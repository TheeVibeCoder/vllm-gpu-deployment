// Raw file-to-GPU transfer benchmark. Build on Linux with nvcc (see README).
// A normal C++ build supports only --cpu-only for an I/O fixture smoke run.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#if defined(__CUDACC__) && defined(__linux__)
#include <cuda_runtime.h>
#include <cufile.h>
#define SSD_GPU_ENABLED 1
#endif

namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t kAlignment = 4096;

std::string system_error(const std::string& action) {
    return action + ": " + std::strerror(errno);
}

struct File {
    int fd = -1;
    explicit File(int descriptor = -1) : fd(descriptor) {}
    ~File() { if (fd >= 0) ::close(fd); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
};

struct Fixture {
    std::string path;
    File file;
    Fixture(const std::string& dir, const std::vector<unsigned char>& expected) {
        std::string pattern = dir + "/ssd-gpu-bench-XXXXXX";
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        file.fd = ::mkstemp(name.data());
        if (file.fd < 0) throw std::runtime_error(system_error("mkstemp"));
        path = name.data();
        try {
            size_t offset = 0;
            while (offset < expected.size()) {
                ssize_t n = ::write(file.fd, expected.data() + offset, expected.size() - offset);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) throw std::runtime_error(system_error("writing fixture"));
                offset += static_cast<size_t>(n);
            }
            if (::fsync(file.fd) != 0) throw std::runtime_error(system_error("fsync fixture"));
        } catch (...) {
            ::unlink(path.c_str());
            throw;
        }
    }
    ~Fixture() { if (!path.empty()) ::unlink(path.c_str()); }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
};

struct AlignedBuffer {
    void* ptr = nullptr;
    explicit AlignedBuffer(size_t size) {
        int err = ::posix_memalign(&ptr, kAlignment, size);
        if (err) throw std::runtime_error("posix_memalign: " + std::string(std::strerror(err)));
    }
    ~AlignedBuffer() { std::free(ptr); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

size_t parse_positive(const char* text, const char* label) {
    if (*text == '-' || *text == '\0') throw std::runtime_error(std::string("invalid ") + label);
    errno = 0;
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || *end || value == 0 || value > std::numeric_limits<size_t>::max())
        throw std::runtime_error(std::string("invalid ") + label);
    return static_cast<size_t>(value);
}

struct Unavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void read_exact(int fd, void* buffer, size_t bytes, bool direct = false) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = ::pread(fd, static_cast<unsigned char*>(buffer) + done, bytes - done,
                            static_cast<off_t>(done));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && direct) throw Unavailable(system_error("O_DIRECT pread"));
        if (n < 0) throw std::runtime_error(system_error("pread"));
        if (n == 0) throw std::runtime_error("short read (EOF)");
        done += static_cast<size_t>(n);
    }
}

void verify(const std::vector<unsigned char>& actual, const std::vector<unsigned char>& expected) {
    if (actual != expected) throw std::runtime_error("GPU/fixture data mismatch");
}

void report(const char* label, std::vector<double> samples, size_t bytes) {
    std::sort(samples.begin(), samples.end());
    double sum = 0;
    for (double us : samples) sum += us;
    double mean = sum / samples.size();
    double p50 = samples[(samples.size() - 1) / 2];
    double p95 = samples[static_cast<size_t>(std::ceil(0.95 * samples.size())) - 1];
    std::cout << std::fixed << std::setprecision(1) << label << ": mean=" << mean
              << " us, p50=" << p50 << " us, p95=" << p95 << " us, throughput="
              << (static_cast<double>(bytes) / mean) << " MB/s (decimal)\n";
}

template <typename Transfer, typename Check>
void run(const char* label, size_t bytes, size_t warmups, size_t iterations,
         Transfer transfer, Check check) {
    std::vector<double> samples;
    samples.reserve(iterations);
    for (size_t i = 0; i < warmups + iterations; ++i) {
        auto start = Clock::now();
        transfer(); // Entire synchronous pread + completed H2D transfer, or synchronous cuFileRead.
        auto end = Clock::now();
        check(); // Data validation stays outside the measured window.
        if (i >= warmups)
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    report(label, samples, bytes);
}

#ifdef SSD_GPU_ENABLED
void cuda_check(cudaError_t status, const char* action) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

struct GpuBuffers {
    void* device = nullptr;
    cudaStream_t stream = nullptr;
    explicit GpuBuffers(size_t bytes) {
        cuda_check(cudaMalloc(&device, bytes), "cudaMalloc");
        try {
            cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
        } catch (...) { cudaFree(device); throw; }
    }
    ~GpuBuffers() {
        cudaStreamDestroy(stream);
        cudaFree(device);
    }
    GpuBuffers(const GpuBuffers&) = delete;
    GpuBuffers& operator=(const GpuBuffers&) = delete;
};

struct PinnedBuffer {
    explicit PinnedBuffer(void* ptr, size_t bytes) : ptr(ptr) {
        cudaError_t status = cudaHostRegister(ptr, bytes, cudaHostRegisterDefault);
        if (status != cudaSuccess)
            throw Unavailable(std::string("cudaHostRegister: ") + cudaGetErrorString(status));
    }
    ~PinnedBuffer() { cudaHostUnregister(ptr); }
    void* ptr;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

void verify_gpu(void* device, std::vector<unsigned char>& actual,
                const std::vector<unsigned char>& expected) {
    cuda_check(cudaMemcpy(actual.data(), device, actual.size(), cudaMemcpyDeviceToHost),
               "cudaMemcpy device-to-host for verification");
    verify(actual, expected);
}

void discard_cache(int fd, size_t bytes) {
    // Advisory only; report that this is not a guaranteed cold read.
    int advice = ::posix_fadvise(fd, 0, static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
    if (advice)
        throw std::runtime_error("posix_fadvise(DONTNEED): " + std::string(std::strerror(advice)));
}

// Failure in a cuFile API is a skipped mode, never an implicit buffered result.
struct Gds {
    bool driver = false;
    bool registered = false;
    CUfileHandle_t handle = nullptr;
    void* device;
    explicit Gds(int fd, void* gpu, size_t bytes) : device(gpu) {
        try {
            CUfileError_t result = cuFileDriverOpen();
            if (result.err != CU_FILE_SUCCESS)
                throw Unavailable(std::string("cuFileDriverOpen: ") + CUFILE_ERRSTR(result.err) +
                                  " (code " + std::to_string(result.err) + ")");
            driver = true;
            CUfileDescr_t desc = {};
            desc.handle.fd = fd;
            desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            result = cuFileHandleRegister(&handle, &desc);
            if (result.err != CU_FILE_SUCCESS)
                throw Unavailable(std::string("cuFileHandleRegister: ") + CUFILE_ERRSTR(result.err) +
                                  " (code " + std::to_string(result.err) + ")");
            result = cuFileBufRegister(device, bytes, 0);
            if (result.err != CU_FILE_SUCCESS)
                throw Unavailable(std::string("cuFileBufRegister: ") + CUFILE_ERRSTR(result.err) +
                                  " (code " + std::to_string(result.err) + ")");
            registered = true;
        } catch (...) { cleanup(); throw; }
    }
    void cleanup() {
        if (registered) cuFileBufDeregister(device);
        if (handle) cuFileHandleDeregister(handle);
        if (driver) cuFileDriverClose();
    }
    ~Gds() { cleanup(); }
    Gds(const Gds&) = delete;
    Gds& operator=(const Gds&) = delete;
};
#endif

} // namespace

int main(int argc, char** argv) {
    try {
        std::string dir, input;
        size_t bytes = 4 * 1024 * 1024, iterations = 20, warmups = 3;
        bool cpu_only = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "Usage: " << argv[0]
                          << " --disk-dir EXISTING_DIRECTORY [--input RAW_KV_FILE] [--bytes N]"
                             " [--iterations N] [--warmups N] [--cpu-only]\n"
                             "--bytes must be a multiple of 4096 (default 4194304)."
                             " --input reads the first N bytes of an existing regular file."
                             " --cpu-only is a buffered I/O fixture smoke run, NOT a GPU benchmark.\n";
                return 0;
            }
            if (arg == "--cpu-only") { cpu_only = true; continue; }
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            if (arg == "--disk-dir") dir = argv[++i];
            else if (arg == "--input") input = argv[++i];
            else if (arg == "--bytes") bytes = parse_positive(argv[++i], "--bytes");
            else if (arg == "--iterations") iterations = parse_positive(argv[++i], "--iterations");
            else if (arg == "--warmups") warmups = parse_positive(argv[++i], "--warmups");
            else throw std::runtime_error("unknown option: " + arg);
        }
        if (dir.empty()) throw std::runtime_error("--disk-dir is required (use the GPU host's local NVMe mount)");
        if (bytes % kAlignment || bytes > static_cast<size_t>(std::numeric_limits<ssize_t>::max()) ||
            iterations > std::numeric_limits<size_t>::max() - warmups)
            throw std::runtime_error("invalid --bytes alignment/size or iteration count");
        struct stat st = {};
        if (::stat(dir.c_str(), &st) || !S_ISDIR(st.st_mode))
            throw std::runtime_error("--disk-dir must be an existing writable directory");
#ifndef SSD_GPU_ENABLED
        if (!cpu_only) throw std::runtime_error("GPU benchmark requires Linux, nvcc, CUDA and cuFile; use --cpu-only to smoke-test file I/O");
#endif
        std::vector<unsigned char> expected(bytes);
        if (!input.empty()) {
            File source(::open(input.c_str(), O_RDONLY));
            if (source.fd < 0) throw std::runtime_error(system_error("open --input"));
            struct stat source_stat = {};
            if (::fstat(source.fd, &source_stat) || !S_ISREG(source_stat.st_mode) ||
                source_stat.st_size < static_cast<off_t>(bytes))
                throw std::runtime_error("--input must be a regular file containing at least --bytes bytes");
            read_exact(source.fd, expected.data(), bytes);
        } else {
            uint32_t state = 0x12345678;
            for (auto& byte : expected) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                byte = static_cast<unsigned char>(state);
            }
        }
        Fixture fixture(dir, expected);
        File buffered(::open(fixture.path.c_str(), O_RDONLY));
        if (buffered.fd < 0) throw std::runtime_error(system_error("open buffered"));
        std::vector<unsigned char> host(bytes);
        std::cout << "Fixture: " << bytes << " bytes; " << iterations << " measured reads after "
                  << warmups << " warmups; source=" << (input.empty() ? "synthetic bytes" : input)
                  << "; path=" << fixture.path << "\n";
        if (cpu_only) {
            std::cout << "CPU-only fixture smoke (no GPU, no SSD-to-VRAM measurement)\n";
            run("buffered pread -> CPU", bytes, warmups, iterations,
                [&] { read_exact(buffered.fd, host.data(), bytes); },
                [&] { verify(host, expected); });
            return 0;
        }
#ifdef SSD_GPU_ENABLED
        int devices = 0;
        cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (!devices) throw std::runtime_error("no CUDA GPU available");
        cudaDeviceProp prop = {};
        cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
        std::cout << "GPU: " << prop.name << "; buffered path requests POSIX_FADV_DONTNEED"
                     " before each read (advisory, not a cold-SSD guarantee)."
                     " Verification copies are not timed.\n";
        GpuBuffers gpu(bytes);
        std::vector<unsigned char> actual(bytes);
        discard_cache(buffered.fd, bytes);
        run("buffered pread + cudaMemcpy (pageable)", bytes, warmups, iterations,
            [&] {
                read_exact(buffered.fd, host.data(), bytes);
                cuda_check(cudaMemcpy(gpu.device, host.data(), bytes, cudaMemcpyHostToDevice),
                           "cudaMemcpy host-to-device");
            }, [&] {
                verify_gpu(gpu.device, actual, expected);
                discard_cache(buffered.fd, bytes);
            });
#ifdef O_DIRECT
        File direct(::open(fixture.path.c_str(), O_RDONLY | O_DIRECT));
        if (direct.fd < 0) {
            std::cout << "O_DIRECT: unavailable (" << system_error("open") << ")\n";
        } else {
            try {
                AlignedBuffer aligned(bytes);
                PinnedBuffer pinned(aligned.ptr, bytes);
                run("O_DIRECT pread + pinned cudaMemcpyAsync", bytes, warmups, iterations,
                    [&] {
                        read_exact(direct.fd, aligned.ptr, bytes, true);
                        cuda_check(cudaMemcpyAsync(gpu.device, aligned.ptr, bytes,
                                                  cudaMemcpyHostToDevice, gpu.stream), "cudaMemcpyAsync");
                        cuda_check(cudaStreamSynchronize(gpu.stream), "cudaStreamSynchronize");
                    }, [&] { verify_gpu(gpu.device, actual, expected); });
            } catch (const Unavailable& error) {
                std::cout << "O_DIRECT: unavailable (" << error.what() << ")\n";
            }
            try {
                Gds gds(direct.fd, gpu.device, bytes);
                run("cuFileRead -> GPU (P2P unverified)", bytes, warmups, iterations,
                    [&] {
                        ssize_t n = cuFileRead(gds.handle, gpu.device, bytes, 0, 0);
                        if (n < 0) throw Unavailable("cuFileRead returned " + std::to_string(n));
                        if (n != static_cast<ssize_t>(bytes))
                            throw std::runtime_error("cuFileRead short read: " + std::to_string(n));
                    }, [&] { verify_gpu(gpu.device, actual, expected); });
            } catch (const Unavailable& error) {
                std::cout << "cuFile: unavailable (" << error.what() << ")\n";
            }
        }
#else
        std::cout << "O_DIRECT and cuFile: unavailable (O_DIRECT not defined)\n";
#endif
        std::cout << "cuFile can transparently use host-memory compatibility mode; this timing"
                     " does not prove NVMe-to-GPU P2P DMA. Check nvidia-fs counters and gdscheck -p"
                     " on the tested host/mount.\n";
#endif
    } catch (const std::exception& error) {
        std::cerr << "benchmark failed: " << error.what() << "\n";
        return 1;
    }
}
