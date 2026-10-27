#define STRATA_TASK_PLANNER_TEST
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include "gpu_tasks.cuh"
using namespace strata::core;
#define CK(x)                                                                                                          \
    do {                                                                                                               \
        auto status = (x);                                                                                             \
        if (status != cudaSuccess) {                                                                                   \
            fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(status));                                               \
            std::exit(2);                                                                                              \
        }                                                                                                              \
    } while (0)
#define REQUIRE(x)                                                                                                     \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #x);                                                     \
            std::exit(3);                                                                                              \
        }                                                                                                              \
    } while (0)
__host__ __device__ uint32_t value(int expert, int word) {
    return uint32_t(expert) * 4099u ^ uint32_t(word) * 747796405u;
}
__global__ void verify_weights(const uint64_t *ptr, const int *expert, int groups, int words, int *errors) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (int64_t)groups * words;
         i += (int64_t)gridDim.x * blockDim.x) {
        int q = int(i / words), j = int(i % words);
        if (((const uint32_t *)ptr[q])[j] != value(expert[q], j))
            atomicAdd(errors, 1);
    }
}
void exercise(int slots, int bytes, int iterations) {
    constexpr int E = 32, cap = 80;
    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    uint32_t *host_weights, *mapped_weights, *clock, *flags;
    uint8_t *cache_data, *stage;
    int32_t *ids, *plan, *counts, *errors, *expected_ids;
    uint64_t *stats, *check_ptr;
    GpuTaskPacket *packet, *device_packet;
    GpuExpertAddress *addresses;
    GpuColdCacheEntry *entries;
    GpuColdCachePlan *cold;
    CK(cudaHostAlloc((void **)&host_weights, (size_t)E * bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostGetDevicePointer((void **)&mapped_weights, host_weights, 0));
    for (int e = 0; e < E; ++e)
        for (int j = 0; j < bytes / 4; ++j)
            host_weights[(size_t)e * (bytes / 4) + j] = value(e, j);
    CK(cudaHostAlloc((void **)&packet, sizeof(*packet), cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostGetDevicePointer((void **)&device_packet, packet, 0));
    std::memset(packet, 0, sizeof(*packet));
    CK(cudaMalloc((void **)&cache_data, (size_t)slots * bytes + 64));
    CK(cudaMemset(cache_data, 0xcd, (size_t)slots * bytes + 64));
    CK(cudaMalloc((void **)&stage, (size_t)16 * bytes + 64));
    CK(cudaMemset(stage, 0xcd, (size_t)16 * bytes + 64));
    CK(cudaMalloc((void **)&ids, cap * 4));
    CK(cudaMalloc((void **)&plan, 4096));
    CK(cudaMalloc((void **)&counts, 16));
    CK(cudaMalloc((void **)&errors, 4));
    CK(cudaMalloc((void **)&flags, 16));
    CK(cudaMalloc((void **)&check_ptr, cap * 8));
    CK(cudaMalloc((void **)&expected_ids, cap * 4));
    CK(cudaMalloc((void **)&addresses, E * sizeof(GpuExpertAddress)));
    CK(cudaMalloc((void **)&entries, slots * sizeof(GpuColdCacheEntry)));
    CK(cudaMalloc((void **)&cold, sizeof(GpuColdCachePlan)));
    CK(cudaMalloc((void **)&clock, 4));
    CK(cudaMalloc((void **)&stats, 48));
    CK(cudaMemset(clock, 0, 4));
    CK(cudaMemset(stats, 0, 48));
    std::vector<GpuExpertAddress> table(E);
    for (int e = 0; e < E; ++e)
        table[e] = {0, 0, (uint64_t)(mapped_weights + (size_t)e * (bytes / 4)), 0};
    std::vector<GpuColdCacheEntry> cache(slots);
    for (int s = 0; s < slots; ++s)
        cache[s] = {-1, 0, (uint64_t)(cache_data + (size_t)s * bytes)};
    CK(cudaMemcpy(addresses, table.data(), table.size() * sizeof(table[0]), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(entries, cache.data(), cache.size() * sizeof(cache[0]), cudaMemcpyHostToDevice));
    GpuTaskPlan p{};
    p.ids = ids;
    p.addresses = addresses;
    p.plan = plan;
    p.host_counts = counts;
    p.packet = device_packet;
    p.flag_a = flags;
    p.flag_b = flags + 1;
    p.n = 20;
    p.k = 10;
    p.experts = E;
    p.cap = cap;
    p.staging_cap = 16;
    p.pcie_num = 256;
    p.sequence = 1;
    p.joint_schedule = 2;
    p.cost = {{100, 600}, {3, 8}, {.5f, 2}, 130, 1.5f, 10000, 1000, 40};
    p.cold_cache = entries;
    p.cold_plan = cold;
    p.cold_clock = clock;
    p.cold_stats = stats;
    p.cold_stage = (uint64_t)stage;
    p.blob_bytes = bytes;
    p.cold_cache_slots = slots;
    const int offset = (4 + cap + 1 + 2 * cap + 1) & ~1;
    auto *cold_sources = (unsigned long long *)(plan + offset) + cap;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    gpu_tasks_plan(p, stream);
    gpu_tasks_fetch_cached(cold_sources, plan + 2, cold, entries, clock, stats, bytes, 64, stream);
    CK(cudaStreamEndCapture(stream, &graph));
    CK(cudaGraphInstantiate(&exec, graph, 0));
    int protected_hit_cases = 0, empty_copy_cases = 0;
    for (int trial = 0; trial < iterations; ++trial) {
        int input[20];
        for (int i = 0; i < 20; ++i)
            input[i] = (trial % 7 == 0 || trial == 2) ? 0 : (trial % 3 == 1 ? i % 10 : (i % 10 + trial) % E);
        if (trial == 0)
            for (int i = 0; i < 20; ++i)
                input[i] = i % 10;
        CK(cudaMemcpyAsync(ids, input, sizeof(input), cudaMemcpyHostToDevice, stream));
        std::vector<GpuColdCacheEntry> previous(slots);
        CK(cudaMemcpy(previous.data(), entries, slots * sizeof(previous[0]), cudaMemcpyDeviceToHost));
        CK(cudaGraphLaunch(exec, stream));
        CK(cudaStreamSynchronize(stream));
        int saved[1024];
        CK(cudaMemcpy(saved, plan, sizeof(saved), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(cache.data(), entries, slots * sizeof(cache[0]), cudaMemcpyDeviceToHost));
        REQUIRE(packet->error == 0 && packet->cpu_groups == 0);
        if (saved[2] == 0)
            ++empty_copy_cases;
        for (int s = 0; s < slots; ++s)
            if (previous[s].expert >= 0 && std::find(input, input + 20, previous[s].expert) != input + 20) {
                REQUIRE(cache[s].expert == previous[s].expert);
                ++protected_hit_cases;
            }
        std::vector<uint64_t> ptrs;
        std::vector<int> expected;
        for (int kind : {0, 1}) {
            int group = 0;
            for (int i = 0; i < 20; ++i)
                if (packet->kind[i] == kind && std::find(input, input + i, input[i]) == input + i) {
                    auto *source = (uint64_t *)(saved + offset) + (kind == 1 ? cap : 0);
                    ptrs.push_back(source[group++]);
                    expected.push_back(input[i]);
                }
        }
        CK(cudaMemcpyAsync(check_ptr, ptrs.data(), ptrs.size() * 8, cudaMemcpyHostToDevice, stream));
        CK(cudaMemcpyAsync(expected_ids, expected.data(), expected.size() * 4, cudaMemcpyHostToDevice, stream));
        CK(cudaMemsetAsync(errors, 0, 4, stream));
        verify_weights<<<64, 256, 0, stream>>>(check_ptr, expected_ids, (int)ptrs.size(), bytes / 4, errors);
        CK(cudaStreamSynchronize(stream));
        int wrong;
        CK(cudaMemcpy(&wrong, errors, 4, cudaMemcpyDeviceToHost));
        REQUIRE(wrong == 0);
    }
    // A temporary hit may neither override a resident helper nor resurrect a skipped tail.
    p.tail_rank = 7;
    table[8].remote = 0x12340;
    CK(cudaMemcpy(addresses, table.data(), table.size() * sizeof(table[0]), cudaMemcpyHostToDevice));
    int route_ids[20];
    for (int i = 0; i < 20; ++i)
        route_ids[i] = i % 10;
    CK(cudaMemcpy(ids, route_ids, sizeof(route_ids), cudaMemcpyHostToDevice));
    for (int cached : {7, 8, 0}) {
        for (auto &e : cache)
            e.expert = -1;
        cache[0].expert = cached;
        CK(cudaMemcpy(entries, cache.data(), cache.size() * sizeof(cache[0]), cudaMemcpyHostToDevice));
        gpu_tasks_plan(p, stream);
        CK(cudaStreamSynchronize(stream));
        REQUIRE(packet->kind[7] == -2 && packet->kind[17] == -2);
        REQUIRE(packet->kind[8] == 2 && packet->kind[18] == 2);
        if (cached == 0)
            REQUIRE(packet->kind[0] == 0 && packet->kind[10] == 0);
    }
    printf("PASS original helper ownership and tail policy preserved\n");
    uint8_t canary[64];
    CK(cudaMemcpy(canary, cache_data + (size_t)slots * bytes, 64, cudaMemcpyDeviceToHost));
    for (auto b : canary)
        REQUIRE(b == 0xcd);
    CK(cudaMemcpy(canary, stage + (size_t)16 * bytes, 64, cudaMemcpyDeviceToHost));
    for (auto b : canary)
        REQUIRE(b == 0xcd);
    REQUIRE(protected_hit_cases > 0 && empty_copy_cases > 0);
    uint64_t totals[6];
    CK(cudaMemcpy(totals, stats, 48, cudaMemcpyDeviceToHost));
    REQUIRE(totals[0] > 0 && totals[2] > 0);
    printf("PASS slots=%d bytes=%d replays=%d protected_hits=%d empty_copy=%d hits=%llu admissions=%llu\n", slots,
           bytes, iterations, protected_hit_cases, empty_copy_cases, (unsigned long long)totals[0],
           (unsigned long long)totals[2]);
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    CK(cudaStreamDestroy(stream));
    for (void *v : {(void *)cache_data, (void *)stage, (void *)ids, (void *)plan, (void *)counts, (void *)errors,
                    (void *)flags, (void *)check_ptr, (void *)expected_ids, (void *)addresses, (void *)entries,
                    (void *)cold, (void *)clock, (void *)stats})
        CK(cudaFree(v));
    CK(cudaFreeHost(host_weights));
    CK(cudaFreeHost(packet));
}
int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        puts("SKIP: CUDA device required");
        return 77;
    }
    CK(cudaSetDevice(argc > 1 ? std::atoi(argv[1]) : 0));
    for (int slots : {1, 2, 4}) {
        exercise(slots, 1024, 140);
        exercise(slots, 2 * 1024 * 1024, 14);
    }
    return 0;
}
