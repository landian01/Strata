#pragma once
#include <cstdint>

namespace strata::core {
// Appended to the helper's one pinned metadata upload. Graphs read this anew
// on every replay, including the mapped aliases and compact-to-final row map.
struct RemoteTransferMeta {
    uint64_t input = 0, output = 0, ready = 0;
    uint32_t sequence = 0, verify = 0;
    int32_t entries = 0;
    int32_t row[80] = {};
};
void remote_reset_ready(uint32_t *ready, int rows, void *stream);
void remote_copy_ready(float *dst, const float *src, int rows, int width, const int32_t *hit_rows, const int32_t *count,
                       const uint32_t *ready, uint32_t sequence, const uint32_t *abort_flag, void *stream);
} // namespace strata::core
