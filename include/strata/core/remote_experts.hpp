#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
class RemoteExpertOpt;
struct RemoteGraphCache;
struct GpuTaskPacket;

/// A profile-filled expert tier on another CUDA device, optionally refreshed
/// between drained decode windows. CUDA0 keeps all
/// dense weights and state; results return through the existing pinned CPU rows.
class RemoteExperts {
public:
    // A layer's stable, first-seen expert order, built by the primary planner.
    // Copied into pinned graph metadata before begin() returns.
    struct RoutePlan {
        int groups = 0, entries = 0;
        int32_t expert[80], slot[80], start[81], row[80];
        RoutePlan() { start[0] = 0; }
    };
    RemoteExperts() = default;
    ~RemoteExperts();
    RemoteExperts(const RemoteExperts&) = delete;
    RemoteExperts& operator=(const RemoteExperts&) = delete;

    /// Initialise the device before the host expert arena registers
    /// tens of GiB of portable mapped memory with CUDA.
    static bool preflight(int device, double& free_gib, std::string& err);
    bool open(int device, int slots, int64_t layers, int64_t experts,
              const std::vector<std::pair<int32_t, int32_t>>& ranked,
              const ExpertCache& primary, ExpertSource& source,
              std::vector<uint8_t>& claimed, std::string& err, bool auto_size = false);
    void close();
    /// Opt-in cache maintenance at a drained verify/primary-adaptation boundary.
    /// Never replaces a slot referenced by a running helper graph. Incoming
    /// experts exclude both primary residency and pending primary promotions.
    bool rebalance_cache(const std::vector<float>& usage, const std::vector<int32_t>& primary,
                         const std::vector<std::pair<int32_t,int32_t>>& pending, int max_swaps,
                         ExpertSource& source, std::string& err);
    uint64_t cache_upload_bytes() const { return cache_upload_bytes_; }

    /// `kind` is the primary verifier's classification (-1 = CPU candidate),
    /// or null on the one-token path. Entries already served on CUDA0 are excluded.
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
               int64_t k, const int32_t* kind, const int32_t* primary_res,
               std::string& err, float* out = nullptr, uint32_t* ready = nullptr, uint32_t sequence = 0, const RoutePlan* plan = nullptr);
    bool owns(int64_t index) const { return owned_[(size_t) index] != 0; }
    /// This helper's cache holds (layer, expert): begin() will take its rows unless the plan gave them away.
    bool holds(int64_t layer, int32_t expert) const { return cache_.slot_of(layer, expert) >= 0; }
    int cache_slot(int64_t layer, int32_t expert) const { return cache_.slot_of(layer, expert); }
    uint64_t cached_address(int64_t layer, int32_t expert) const {
        const int slot = cache_.slot_of(layer, expert);
        return slot < 0 ? 0 : (uint64_t) cache_.device_slot(slot);
    }
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    bool prepare_gpu_window(GpuTaskPacket*, int lb, int le, int nt, const float* x, float* out,
                            uint32_t* ready, uint32_t* abort_flag, std::string& err);
    bool launch_gpu_window(int nt, std::string& err);
    bool finish_gpu_window(int lb, int le, std::string& err);
#endif
    uint64_t cold_address(int64_t layer, int32_t expert) const {
        const size_t i=(size_t)layer*n_expert_+expert;
        return i<cold_map_.size()?cold_map_[i]:0;
    }
    bool optimized_decode() const { return remote_opt_ != nullptr; }
    bool direct_output() const { return direct_active_; }
    int device() const { return device_; }
    void* stream_handle() const { return (void*) stream_; }
    bool finish(float* out, std::string& err);
    int64_t resident() const { return cache_.resident(); }
    int64_t computed() const { return computed_; }
    int64_t launched_layers() const { return launched_layers_; }
    double gib() const { return cache_.gib(); }
    uint64_t returned_bytes() const { return returned_bytes_; }
    uint64_t full_row_bytes() const { return full_row_bytes_; }
    /// host time spent in begin() (staging + launches) and in finish() (waiting for this GPU), cumulative
    double ms_begin() const { return ms_begin_; }
    double ms_wait() const { return ms_wait_; }

    /// STRATA_PREFILL_HELPER_SOURCE (opt-in): copy up to `n` cached blobs into the host buffers `dst[i]`
    /// on this helper's own stream and wait.  `hit[i]` is set where this helper held (layer, expert); the
    /// bytes are the ones open() filled from the same source, so the prompt path gets identical bytes.
    /// Misses are left to the caller.  False with `err` on a CUDA failure; the hits copied so far are
    /// complete (the stream is drained), so the caller falls back for the rest.  Safe from several threads
    /// (one stream, one lock): the prompt path's stager is threaded.
    bool copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n,
                           uint8_t* hit, std::string& err);

private:
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    cudaGraphExec_t task_exec_[9] = {};
    GpuTaskPacket* task_packets_ = nullptr;
    uint64_t task_launches_ = 0;
    cudaStream_t cold_stream_ = nullptr;
    cudaEvent_t cold_fork_ = nullptr;
    std::vector<cudaEvent_t> cold_done_;
    void* cold_meta_ = nullptr;
    uint8_t* cold_weights_ = nullptr;
    unsigned long long* cold_source_ = nullptr;
    int32_t* cold_counts_ = nullptr;
    size_t cold_stride_ = 0;
    int cold_cap_ = 0;
#endif
    bool enqueue(int64_t layer, int64_t n_tok, int64_t cap_groups, int64_t cap_entries, std::string& err,
                 bool direct_output = false);
    bool copy_output(int64_t cap_entries, std::string& err);
    bool copy_input(int64_t n_tok, std::string& err);
    friend class RemoteExpertOpt;
    RemoteExpertOpt* remote_opt_ = nullptr;
    int device_ = -1;
    int64_t n_expert_ = 0;
    int32_t groups_ = 0;
    int64_t computed_ = 0;
    uint64_t cache_upload_bytes_ = 0, cache_swap_count_ = 0, cache_rebalances_ = 0;
    int64_t launched_layers_ = 0;
    uint64_t returned_bytes_ = 0;
    uint64_t full_row_bytes_ = 0;
    double ms_begin_ = 0, ms_wait_ = 0;
    void* keepalive_ = nullptr;  // optional request-triggered CUDA clock keepalive
    RemoteGraphCache* graphs_ = nullptr;
    bool group_buffer_ = false, group_verify_ = false;
    uint32_t group_epoch_ = 0;
    uint64_t group_checks_ = 0;
    std::vector<uint32_t> expert_epoch_;
    std::vector<int32_t> expert_group_, group_size_, group_cursor_;
    ExpertCache cache_;
    cudaStream_t stream_ = nullptr;
    cudaStream_t read_stream_ = nullptr;   ///< STRATA_PREFILL_HELPER_SOURCE: cache -> host copies
    std::mutex read_mu_;                   ///< one cache reader at a time (the stager is threaded)
    float* h_x_ = nullptr;
    float* h_out_ = nullptr;
    void* h_meta_ = nullptr;
    void* h_meta_base_ = nullptr;
    size_t meta_stride_ = 0;
    int meta_slot_ = 0;
    bool separate_ready_ = false, async_active_ = false;
    float* d_x_ = nullptr;
    float* z_x_ = nullptr;     ///< h_x_ as the helper GPU sees it (zero-copy: no input copy per layer)
    float* z_out_ = nullptr;   ///< h_out_ as the helper GPU sees it (zero-copy: no result copy)
    bool zero_copy_ = false;
    bool output_kernel_ = false, output_verify_ = false;
    bool direct_io_ = false, direct_input_ = false, direct_active_ = false;
    void* d_transfer_ = nullptr;
    uint64_t direct_calls_ = 0;
    // A verifier uses a small fixed set of mapped input/output slice addresses.
    std::vector<std::pair<const void*, void*>> mapped_aliases_;
    uint64_t output_calls_ = 0, output_checks_ = 0;
    std::vector<float> output_reference_;
    float* d_out_ = nullptr;
    uint8_t* d_q8_ = nullptr;
    float* d_scales_ = nullptr;
    float* d_xr_ = nullptr;                 // APR: the Hadamard-rotated expert input (CAP x H)
    const float* d_sx_ = nullptr;           // APR: device sign vectors (x: H, h: FF); nullptr = identity
    const float* d_sh_ = nullptr;
    void* d_scratch_ = nullptr;
    void* d_meta_ = nullptr;  ///< one contiguous upload of grouped indices, instead of five small copies
    int32_t* d_start_ = nullptr;
    int32_t* d_dst_ = nullptr;
    int32_t* d_tok_ = nullptr;
    int32_t* d_count_ = nullptr;
    unsigned long long* d_ptr_ = nullptr;
    std::vector<uint64_t> cold_map_;
    uint64_t cold_map_base_ = 0, cold_map_chunk_ = 0;
    std::vector<std::pair<uint64_t,uint64_t>> cold_ranges_;
    std::vector<uint8_t> owned_;
    std::vector<uint8_t> layers_present_;
    std::vector<int32_t> group_of_, group_id_;
    std::vector<int32_t> start_, dst_, tok_, original_row_;
    std::vector<unsigned long long> ptr_;
};

/// STRATA_PREFILL_HELPER_SOURCE (opt-in): the prompt path's view of the expert tiers, where a blob a helper
/// GPU already holds in its VRAM cache is copied from that cache instead of being read from the files again.
/// Everything else is forwarded to `fallback` unchanged, so its batched unbuffered reads, its pinned RAM copy
/// and the prompt path's own decisions keep their meaning.  Nothing is cached here: every call asks the
/// helpers, so an entry an adaptive tier moves between requests is a miss (a file read), never a wrong blob.
///
/// The serve loop builds this only for a serial request path (no batch slots, no layer split), while the
/// helpers are idle: during a prompt read the helper GPUs compute nothing, so their caches can be read
/// without ordering the copies against their kernel streams.
class RemotePrefillSource final : public ExpertSource {
public:
    RemotePrefillSource(ExpertSource& fallback, const std::vector<RemoteExperts*>& helpers)
        : fallback_(fallback), helpers_(helpers) {}

    /// Whether any helper's VRAM cache holds (layer, expert) right now.
    bool from_helper(int64_t layer, int64_t expert) const {
        for (const RemoteExperts* h : helpers_) if (h->holds(layer, (int32_t) expert)) return true;
        return false;
    }
    /// Blobs / bytes this source has taken from the helpers so far (the per-request evidence line).
    uint64_t helper_blobs() const { return helper_blobs_.load(std::memory_order_relaxed); }
    uint64_t helper_bytes() const { return helper_bytes_.load(std::memory_order_relaxed); }

    const uint8_t* blob(int64_t layer, int64_t expert) override { return fallback_.blob(layer, expert); }
    const uint8_t* blob_stable(int64_t layer, int64_t expert) override {
        return fallback_.blob_stable(layer, expert);
    }
    int blob_parts(int64_t layer, int64_t expert, BlobPart* out) override {
        return from_helper(layer, expert) ? 0 : fallback_.blob_parts(layer, expert, out);
    }
    int64_t reads() const override { return fallback_.reads(); }
    bool pinned(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? false : fallback_.pinned(layer, expert);
    }
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override {
        fallback_.begin_layer(layer, ids, k);
    }
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? nullptr : fallback_.device_alias(layer, expert);
    }
    void prefetch(int64_t layer, int64_t expert) override { fallback_.prefetch(layer, expert); }
    uint64_t release(int64_t layer, int64_t expert) override {
        return from_helper(layer, expert) ? 0 : fallback_.release(layer, expert);
    }
    bool pcie_layer(int64_t layer) const override { return fallback_.pcie_layer(layer); }
    /// True where a helper holds the blob: the prompt path's stager must then take the copy_blob(s) route,
    /// which is where the cache-to-host copy happens.  Everywhere else the fallback answers as it always did.
    bool transient(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? true : fallback_.transient(layer, expert);
    }
    double cached_share(int64_t samples) const override { return fallback_.cached_share(samples); }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) override;
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override {
        fallback_.prefetch(layer, experts, n);
    }
    void warm(int64_t layer, const int64_t* experts, int64_t n) override {
        fallback_.warm(layer, experts, n);
    }
    bool warms() const override { return fallback_.warms(); }
    bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const override {
        return fallback_.advise_pairs(pairs, n);
    }

private:
    void warn(const std::string& err);
    ExpertSource& fallback_;
    std::vector<RemoteExperts*> helpers_;
    std::atomic<uint64_t> helper_blobs_{0}, helper_bytes_{0};
    std::atomic<bool> warned_{false};
};
} // namespace strata::core
