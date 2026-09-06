#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pocket {

// Host-side bookkeeping for a paged KV cache.
//
// The contiguous arena reserves `max_context` tokens per slot at construction,
// so a batch of short requests at a large context still pays for the worst case.
// This pool owns a fixed number of fixed-size blocks instead and hands them out
// as tokens actually arrive, which is what lets batch size scale with real token
// use rather than with `max_batch_size * max_context`.
//
// Only the block *indices* live here. The device memory is one arena per layer
// owned by the engine, and a block id is an offset into it, so this class has no
// device dependency and is testable without a model.
//
// Blocks are refcounted even though nothing shares them yet: prefix sharing
// across sequences needs exactly this counter, and retrofitting it later would
// mean revisiting every free site.
class QwenBlockPool {
public:
    static constexpr int kInvalidBlock = -1;

    // `block_size` is tokens per block; `num_blocks` is the pool extent. Both
    // must be positive.
    QwenBlockPool(int num_blocks, int block_size);

    // Takes `count` blocks, or returns empty when the pool cannot satisfy the
    // whole request. Partial allocation is deliberately not offered: a sequence
    // that gets some of the blocks it needs cannot run, and the caller would
    // have to hand them straight back. Failure is backpressure, not an error,
    // so it does not throw.
    std::vector<int> allocate(int count);

    // Drops one reference per id and returns any block whose count reaches zero
    // to the free list. Ignores kInvalidBlock so a caller can free a partially
    // built table. Throws on an out-of-range or already-free id, which is a
    // bookkeeping bug rather than a runtime condition.
    void free(const std::vector<int>& block_ids);

    // Adds a reference, for a block that a second sequence begins to share.
    void retain(int block_id);

    int refcount(int block_id) const;

    int free_blocks() const { return static_cast<int>(free_list_.size()); }
    int total_blocks() const { return num_blocks_; }
    int block_size() const { return block_size_; }

    // Blocks needed to hold `tokens`, i.e. the ceiling division. A sequence
    // holding exactly `block_size` tokens needs one block, and one more token
    // needs two.
    int blocks_for_tokens(int tokens) const;

private:
    void validate(int block_id) const;

    int num_blocks_;
    int block_size_;
    // Free ids, taken and returned at the back. LIFO reuse keeps a just-freed
    // block hot in cache rather than cycling through the whole pool.
    std::vector<int> free_list_;
    std::vector<int> refcounts_;
};

}  // namespace pocket
