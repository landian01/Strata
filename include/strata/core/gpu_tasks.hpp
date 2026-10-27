#pragma once
#include "strata/core/remote_transfer.hpp"
#include <cstdint>

namespace strata::core {
constexpr int kGpuTaskRows = 80;
// Same layout as the legacy helper's single metadata allocation.
struct GpuRemoteMeta {
    unsigned long long ptr[80];
    int32_t start[81], dst[80], tok[80], count;
};
struct GpuExpertAddress {
    uint64_t primary, remote, mapped, helper_mapped;
};
// Persistent secondary slots belong to one layer. A tag becomes valid only
// after its copy completes; slots read by this window are never overwritten.
struct GpuColdCacheEntry {
    int32_t expert;
    uint32_t last_use;
    uint64_t pointer;
};
struct GpuColdCachePlan {
    uint64_t destination[kGpuTaskRows];
    int32_t slot[kGpuTaskRows], expert[kGpuTaskRows];
};
// Costs are microseconds: measured copy bandwidth and configurable compute/
// return estimates. The profiler, not this model, establishes actual speed.
// Each layer uses its blob size, resident queue and actual routed row counts.
struct GpuScheduleCost {
    float fetch[2], group[2], row[2], helper_start, helper_return;
    float cpu_group = 125.f, cpu_row = 2.f, cpu_start = 20.f;
    // Full-memory-stream pair penalties, weighted by predicted overlap and
    // CPU memory duty. cpu_ram is one blob / measured shared RAM bandwidth.
    float cpu_ram = 0.f, main_cpu = 0.f, helper_cpu = 0.f;
    float cpu_main = 0.f, cpu_helper = 0.f, gpu_cross = 0.f;
};
// One immutable packet per layer per window. Only ready is reset between windows,
// after both GPUs have drained. Output/input addresses belong to the helper GPU.
struct alignas(64) GpuTaskPacket {
    GpuRemoteMeta remote, cold;
    RemoteTransferMeta transfer;
    int32_t kind[80], cpu_group[80], cpu_expert[80];
    int32_t cpu_groups, skipped, pcie_groups, error;
    float predicted_us[3];
    int32_t reuse_groups, reuse_entries;
    uint32_t ready;
};
struct GpuTaskPlan {
    const int32_t *ids;
    const GpuExpertAddress *addresses;
    int32_t *plan;
    int32_t *host_counts;
    GpuTaskPacket *packet;
    uint32_t *flag_a, *flag_b;
    int n, k, experts, cap, staging_cap, pcie_num, tail_rank, min_one;
    uint32_t sequence;
    int helper_cold_cap = 0;
    GpuScheduleCost cost{};
    int joint_schedule = 0;
    GpuColdCacheEntry *cold_cache = nullptr;
    GpuColdCachePlan *cold_plan = nullptr;
    uint32_t *cold_clock = nullptr;
    uint64_t *cold_stats = nullptr;
    uint64_t cold_stage = 0, blob_bytes = 0;
    int cold_cache_slots = 0;
};
void gpu_tasks_plan(const GpuTaskPlan &plan, void *stream);
void gpu_tasks_fetch_cached(unsigned long long *sources, const int32_t *count, const GpuColdCachePlan *plan,
                            GpuColdCacheEntry *cache, const uint32_t *clock, uint64_t *stats, int64_t blob_bytes,
                            int blocks, void *stream);
void gpu_tasks_load(const GpuTaskPacket *packet, void *meta, const uint32_t *abort_flag, uint32_t sequence,
                    void *stream, void *cold_meta = nullptr);
} // namespace strata::core
