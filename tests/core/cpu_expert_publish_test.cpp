#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include "strata/core/remote_transfer.hpp"
#define CK(x)                                                                                                          \
    do {                                                                                                               \
        auto e = (x);                                                                                                  \
        if (e != cudaSuccess) {                                                                                        \
            std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e));                                               \
            std::exit(10);                                                                                             \
        }                                                                                                              \
    } while (0)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
namespace cpu = strata::kernels::cpu;
int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: cpu_expert_publish_test <Swift/Qwen split GGUF>\n");
        return 2;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    strata::GgufModel model(strata::gguf_split_paths(argv[1]));
    cpu::set_host_core(cpu::HostCore::Last);
    cpu::ExpertPool pool(5, true, true);
    std::printf("%s\n", pool.placement_line().c_str());
    const bool verify = std::getenv("BENCH_READY_VERIFY") != nullptr;
    float *out, *mapped_out, *gpu_out;
    uint32_t *ready, *mapped_ready;
    int32_t *hits, *count;
    cudaStream_t stream;
    CK(cudaHostAlloc(&out, (80 * cpu::H + 8) * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped));
    CK(cudaHostAlloc(&ready, 81 * sizeof(uint32_t), cudaHostAllocPortable | cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer(&mapped_out, out, 0));
    CK(cudaHostGetDevicePointer(&mapped_ready, ready, 0));
    CK(cudaMalloc(&gpu_out, (80 * cpu::H + 8) * sizeof(float)));
    CK(cudaMalloc(&hits, 80 * sizeof(int32_t)));
    CK(cudaMalloc(&count, 4));
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CK(cudaMemsetAsync(count, 0, 4, stream));
    CK(cudaStreamSynchronize(stream));
    std::vector<float> fetched(80 * cpu::H + 8);
    int checks = 0;
    for (int layer : {3, 20, 47}) {
        const strata::TensorInfo *tensors[3]{};
        const uint8_t *data[3]{};
        const char *roles[3] = {"gate", "up", "down"};
        for (int r = 0; r < 3; ++r) {
            size_t shard = 0;
            tensors[r] = model.find("blk." + std::to_string(layer) + ".ffn_" + roles[r] + "_exps.weight", &shard);
            if (!tensors[r])
                return 3;
            data[r] = model.shard(shard).tensor_data(*tensors[r]);
        }
        cpu::NativeFmt fmt;
        std::string error;
        if (!cpu::native_fmt((int)tensors[0]->type, (int)tensors[2]->type, cpu::H, cpu::FF, fmt, error))
            return 4;
        std::vector<uint8_t> blobs(80 * fmt.bytes);
        for (int g = 0; g < 80; ++g) {
            size_t e = (g * 37 + 5) % 256;
            auto b = blobs.data() + g * fmt.bytes;
            std::memcpy(b, data[0] + e * fmt.up_off, fmt.up_off);
            std::memcpy(b + fmt.up_off, data[1] + e * fmt.up_off, fmt.up_off);
            std::memcpy(b + fmt.down_off, data[2] + e * (fmt.bytes - fmt.down_off), fmt.bytes - fmt.down_off);
        }
        std::mt19937 rng(17 + layer);
        std::normal_distribution<float> normal;
        std::vector<float> x(cpu::MAXT * cpu::H);
        for (auto &v : x)
            v = normal(rng);
        std::vector<uint8_t> quant(cpu::MAXT * cpu::kNativeActBytes);
        std::vector<cpu::ActQ> q2(cpu::MAXT);
        for (int t = 0; t < cpu::MAXT; ++t) {
            cpu::native_quant_act(fmt, x.data() + t * cpu::H, quant.data() + t * cpu::kNativeActBytes);
            cpu::act_quant_any(x.data() + t * cpu::H, cpu::H, q2[t]);
        }
        for (int groups : {1, 4, 8, 16, 32, 80}) {
            std::vector<cpu::ExpertJobMulti> jobs(groups);
            int entries = 0;
            for (int g = 0; g < groups; ++g) {
                auto &j = jobs[g];
                j.blob = blobs.data() + g * fmt.bytes;
                j.nt = groups == 80 ? 1 : groups <= 8 ? 1 + (g * 3 + 7) % 8 : 1 + g % 3;
                j.ready_sequence = 13;
                entries += j.nt;
                for (int t = 0; t < j.nt; ++t) {
                    j.nact[t] = quant.data() + t * cpu::kNativeActBytes;
                    j.act[t] = &q2[t];
                    j.out[t] = out + (entries - j.nt + t) * cpu::H;
                    j.row_ready[t] = ready + entries - j.nt + t;
                }
            }
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            strata::core::remote_copy_ready(gpu_out, mapped_out, entries, cpu::H, hits, count, mapped_ready, 13,
                                            mapped_ready + 80, stream);
            CK(cudaStreamEndCapture(stream, &graph));
            CK(cudaGraphInstantiate(&exec, graph, 0));
            std::vector<double> us;
            for (int it = -20; it < 160; ++it) {
                if (verify) {
                    for (auto &v : x)
                        v = normal(rng);
                    for (int t = 0; t < cpu::MAXT; ++t) {
                        cpu::native_quant_act(fmt, x.data() + t * cpu::H, quant.data() + t * cpu::kNativeActBytes);
                        cpu::act_quant_any(x.data() + t * cpu::H, cpu::H, q2[t]);
                    }
                }
                for (int j = 0; j < 80; ++j)
                    ready[j] = 13;
                ready[80] = 0;
                std::fill(out, out + 80 * cpu::H + 8, -777.f);
                strata::core::remote_reset_ready(mapped_ready, entries, stream);
                CK(cudaMemsetAsync(gpu_out, 0, (80 * cpu::H + 8) * sizeof(float), stream));
                CK(cudaStreamSynchronize(stream));
                CK(cudaGraphLaunch(exec, stream));
                auto flush = cudaStreamQuery(stream);
                if (flush != cudaSuccess && flush != cudaErrorNotReady)
                    CK(flush);
                auto start = std::chrono::steady_clock::now();
                pool.run_split_multi_native(fmt, jobs.data(), groups);
                // Batch publication is the old-path control and a harmless
                // duplicate for the per-expert publisher.
                for (int row = 0; row < entries; ++row)
                    std::atomic_ref<uint32_t>(ready[row]).store(13, std::memory_order_release);
                CK(cudaStreamSynchronize(stream));
                auto stop = std::chrono::steady_clock::now();
                CK(cudaMemcpy(fetched.data(), gpu_out, (80 * cpu::H + 8) * sizeof(float), cudaMemcpyDeviceToHost));
                if (std::memcmp(fetched.data(), out, entries * cpu::H * sizeof(float)) != 0) {
                    std::fprintf(stderr, "early publication mismatch layer %d groups %d iteration %d\n", layer, groups,
                                 it);
                    return 11;
                }
                for (int j = entries * cpu::H; j < 80 * cpu::H + 8; ++j)
                    if (fetched[j] != 0.f || out[j] != -777.f)
                        return 12;
                ++checks;
                if (it >= 0)
                    us.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
            }
            std::sort(us.begin(), us.end());
            std::printf("layer=%d groups=%d entries=%d p50_us=%.3f p95_us=%.3f\n", layer, groups, entries, us[80],
                        us[152]);
            CK(cudaGraphExecDestroy(exec));
            CK(cudaGraphDestroy(graph));
        }
    }
    CK(cudaStreamDestroy(stream));
    CK(cudaFree(gpu_out));
    CK(cudaFree(hits));
    CK(cudaFree(count));
    CK(cudaFreeHost(out));
    CK(cudaFreeHost(ready));
    std::printf("PASS %d GPU consumer checks, stale generation resets and guards\n", checks);
    return 0;
}
