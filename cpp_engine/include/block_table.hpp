#pragma once

#include "block_pool.hpp"

#include <cstdint>
#include <vector>

namespace pocket {

// Logical-to-physical position mapping for the paged KV cache, one row per slot.
//
// Row `slot` holds the block ids backing that sequence in logical order, so
// logical position `pos` lives in block `rows[slot][pos / block_size]` at offset
// `pos % block_size` within it. This is the paged replacement for
// `slot * max_context * kv_heads * head_dim`, and it is the only place the
// translation is written down on the host side.
//
// The device mirror is a dense `[max_slots, max_blocks_per_seq]` int32 image of
// the same rows, which is what the kernels index. It is kept as host memory here
// and uploaded by the engine, so this class stays free of device calls and can
// be tested without a GPU.
class BlockTable {
public:
    BlockTable(BlockPool* pool, int max_slots, int max_context);

    // Ensures `slot` is backed to at least `tokens` logical positions, taking
    // blocks from the pool as needed. Returns false without changing anything
    // when the pool cannot cover the growth, so the caller can leave the request
    // waiting instead of failing it. Shrinking is not supported: a sequence only
    // ever grows until it is released.
    bool ensure_capacity(int slot, int tokens);

    // Returns every block held by `slot` to the pool and clears the row.
    void release(int slot);

    void release_all();

    // Physical element offset of a logical position within the layer arena,
    // measured in cache elements rather than bytes so it composes with the
    // existing `+ slot_offset` pointer arithmetic. `kv_stride` is
    // `kv_heads * head_dim`, the elements per token.
    size_t element_offset(int slot, int position, size_t kv_stride) const;

    // Blocks currently backing `slot`.
    const std::vector<int>& row(int slot) const;

    // Logical positions currently backed for `slot`, i.e. blocks * block_size.
    int capacity_tokens(int slot) const;

    // Flat `[max_slots, max_blocks_per_seq]` image for upload, padded with
    // kInvalidBlock. Rebuilt only when a row changed since the last call, since
    // a decode step that crosses no block boundary should not pay for an
    // upload. `dirty()` reports whether that is the case.
    const std::vector<int32_t>& device_image();
    bool dirty() const { return dirty_; }

    int max_blocks_per_seq() const { return max_blocks_per_seq_; }
    int max_slots() const { return max_slots_; }
    int block_size() const { return pool_->block_size(); }

private:
    void validate_slot(int slot) const;

    BlockPool* pool_;
    int max_slots_;
    int max_context_;
    int max_blocks_per_seq_;
    std::vector<std::vector<int>> rows_;
    std::vector<int32_t> image_;
    bool dirty_ = true;
};

}  // namespace pocket
