#include "strata/core/pinned.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}

__global__ void gather(const unsigned long long *input, unsigned long long *output, size_t pages) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < pages)
        output[i] = input[i * (4096 / 8)];
}

int main(int argc, char **argv) {
    try {
        int devices = 0;
        check(cudaGetDeviceCount(&devices));
        if (devices < 2) {
            std::puts("SKIP: two CUDA devices required");
            return 77;
        }
        for (int d = 0; d < 2; ++d) {
            check(cudaSetDevice(d));
            check(cudaFree(nullptr));
        }
        check(cudaSetDevice(0));
        _putenv_s("STRATA_ARENA_PIN_GIB", "0");
        _putenv_s("STRATA_ARENA_VMM", "1");
        const uint64_t bytes = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (1ull << 30) + 123 * 4096;
        const size_t pages = size_t(bytes / 4096);
        for (int iteration = 0; iteration < 3; ++iteration) {
            {
                strata::core::PinnedArena a(bytes, 1ull << 30);
                if (!a.valid() || a.vmm_device != 0 || a.registered_bytes != bytes)
                    throw std::runtime_error(a.note);
                for (size_t i = 0; i < pages; ++i)
                    ((unsigned long long *)a.data())[i * (4096 / 8)] = 0x1234567800000000ull + i + iteration;
                unsigned long long *output = nullptr;
                check(cudaMalloc(&output, pages * 8));
                gather<<<unsigned((pages + 255) / 256), 256>>>((const unsigned long long *)a.data(), output, pages);
                check(cudaGetLastError());
                std::vector<unsigned long long> got(pages);
                check(cudaMemcpy(got.data(), output, pages * 8, cudaMemcpyDeviceToHost));
                check(cudaFree(output));
                for (size_t i = 0; i < pages; ++i)
                    if (got[i] != 0x1234567800000000ull + i + iteration)
                        throw std::runtime_error("GPU mapped-read mismatch");
                // Both cards must still accept VRAM allocations; copy across a VMM chunk boundary to the helper.
                if (bytes > (1ull << 30) + 65536) {
                    const size_t n = 131072;
                    const size_t off = (1ull << 30) - 65536;
                    std::vector<unsigned char> input(n), back(n);
                    for (size_t i = 0; i < n; ++i)
                        input[i] = (unsigned char)(i * 13 + iteration);
                    std::memcpy(a.data() + off, input.data(), n);
                    check(cudaSetDevice(1));
                    unsigned char *gpu = nullptr;
                    check(cudaMalloc(&gpu, n));
                    check(cudaMemcpy(gpu, a.data() + off, n, cudaMemcpyHostToDevice));
                    check(cudaMemcpy(back.data(), gpu, n, cudaMemcpyDeviceToHost));
                    check(cudaFree(gpu));
                    if (input != back)
                        throw std::runtime_error("helper H2D mismatch");
                    check(cudaSetDevice(0));
                }
                std::printf("iteration %d: %zu pages verified, %zu chunks, helper H2D PASS\n", iteration, pages,
                            a.vmm_chunks.size());
            }
            for (int d = 0; d < 2; ++d) {
                check(cudaSetDevice(d));
                size_t free = 0, total = 0;
                check(cudaMemGetInfo(&free, &total));
                void *small = nullptr;
                check(cudaMalloc(&small, 4096));
                check(cudaFree(small));
            }
            check(cudaSetDevice(0));
        }
        _putenv_s("STRATA_ARENA_PIN_GIB", "1");
        {
            strata::core::PinnedArena a(4096, 4096);
            if (a.valid())
                throw std::runtime_error("conflicting pin cap accepted");
        }
        _putenv_s("STRATA_ARENA_PIN_GIB", "0");
        _putenv_s("STRATA_ARENA_VMM", "0");
        {
            strata::core::PinnedArena a(4096, 4096);
            if (!a.valid() || a.vmm_device >= 0)
                throw std::runtime_error("ordinary allocation changed");
        }
        std::puts("PASS: mapped reads, helper DMA, three allocation lifecycles, option guard, ordinary fallback");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
