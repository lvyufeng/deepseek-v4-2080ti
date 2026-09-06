// Host-side checks for the paged KV block pool and block table. No model and no
// device: the point is that the logical-to-physical translation and the free
// list are exercised without a GPU, so a translation bug is caught here rather
// than as a divergence in a 64-layer parity run.

#include "qwen_block_pool.hpp"
#include "qwen_block_table.hpp"

#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) return;
    std::printf("[FAIL] %s\n", what.c_str());
    ++failures;
}

// Returns true when `call` threw, so a bookkeeping bug is asserted to be loud
// rather than silently accepted.
template <typename Fn>
bool throws(Fn&& call) {
    try {
        call();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void test_pool_basics() {
    pocket::QwenBlockPool pool(8, 16);
    check(pool.total_blocks() == 8, "pool total_blocks");
    check(pool.free_blocks() == 8, "fresh pool is fully free");
    check(pool.block_size() == 16, "pool block_size");

    check(pool.blocks_for_tokens(0) == 0, "0 tokens needs no block");
    check(pool.blocks_for_tokens(1) == 1, "1 token needs one block");
    check(pool.blocks_for_tokens(16) == 1, "a full block is one block");
    check(pool.blocks_for_tokens(17) == 2, "one token past a block needs two");

    std::vector<int> first = pool.allocate(3);
    check(first.size() == 3, "allocate(3) returns three blocks");
    check(pool.free_blocks() == 5, "allocation reduces the free count");
    for (const int block : first) {
        check(pool.refcount(block) == 1, "a fresh block has refcount 1");
    }

    pool.free(first);
    check(pool.free_blocks() == 8, "free returns every block");
}

void test_pool_exhaustion() {
    pocket::QwenBlockPool pool(4, 16);
    std::vector<int> all = pool.allocate(4);
    check(all.size() == 4, "the pool can be drained exactly");
    check(pool.free_blocks() == 0, "drained pool has nothing free");

    // Over-request must not partially allocate: a caller that got 2 of 3 blocks
    // would have to hand them back, and a leak here would be invisible.
    check(pool.allocate(1).empty(), "an exhausted pool returns empty");
    check(pool.free_blocks() == 0, "a failed allocation takes nothing");

    pool.free(all);
    check(pool.free_blocks() == 4, "the pool recovers fully");

    std::vector<int> some = pool.allocate(3);
    check(pool.allocate(2).empty(), "an over-request returns empty");
    check(pool.free_blocks() == 1,
          "a failed over-request leaves the free list untouched");
    pool.free(some);
}

void test_pool_refcounts() {
    pocket::QwenBlockPool pool(4, 16);
    std::vector<int> blocks = pool.allocate(1);
    const int block = blocks[0];

    pool.retain(block);
    check(pool.refcount(block) == 2, "retain raises the refcount");
    pool.free({block});
    check(pool.refcount(block) == 1, "one free of a shared block keeps it live");
    check(pool.free_blocks() == 3, "a still-referenced block is not free");
    pool.free({block});
    check(pool.refcount(block) == 0, "the last free drops the refcount to 0");
    check(pool.free_blocks() == 4, "the last free returns the block");

    check(throws([&] { pool.free({block}); }),
          "freeing an already-free block throws");
    check(throws([&] { pool.retain(block); }),
          "retaining a free block throws");
    check(throws([&] { pool.refcount(99); }), "an out-of-range id throws");
    check(throws([&] { pocket::QwenBlockPool(0, 16); }),
          "a zero-block pool throws");
    check(throws([&] { pocket::QwenBlockPool(4, 0); }),
          "a zero-size block throws");

    // kInvalidBlock is skipped so a caller can free a partially built row.
    pool.free({pocket::QwenBlockPool::kInvalidBlock});
    check(pool.free_blocks() == 4, "freeing kInvalidBlock is a no-op");
}

void test_table_growth() {
    pocket::QwenBlockPool pool(16, 16);
    pocket::QwenBlockTable table(&pool, 4, 256);
    check(table.max_blocks_per_seq() == 16, "max_blocks_per_seq from context");
    check(table.capacity_tokens(0) == 0, "a fresh slot backs no tokens");

    check(table.ensure_capacity(0, 1), "one token is backed");
    check(table.capacity_tokens(0) == 16, "one block backs a whole block");
    check(pool.free_blocks() == 15, "growth takes from the pool");

    // Already covered: no new block, and no pool traffic.
    check(table.ensure_capacity(0, 16), "a full block needs no growth");
    check(pool.free_blocks() == 15, "re-covering the same tokens is free");

    check(table.ensure_capacity(0, 17), "crossing a boundary grows");
    check(table.capacity_tokens(0) == 32, "the row is now two blocks");
    check(pool.free_blocks() == 14, "the boundary took one more block");

    table.release(0);
    check(table.capacity_tokens(0) == 0, "release empties the row");
    check(pool.free_blocks() == 16, "release returns every block");
}

void test_table_translation() {
    pocket::QwenBlockPool pool(8, 4);
    pocket::QwenBlockTable table(&pool, 2, 32);
    const size_t kv_stride = 8;  // kv_heads * head_dim, small enough to check

    check(table.ensure_capacity(0, 12), "slot 0 backed to 12 tokens");
    const std::vector<int>& row = table.row(0);
    check(row.size() == 3, "12 tokens over block_size 4 is three blocks");

    // Position p lives at (block[p / 4] * 4 + p % 4) * kv_stride. Spot-check
    // both ends of each block rather than trusting the first one.
    for (int position = 0; position < 12; ++position) {
        const int block = row[static_cast<size_t>(position / 4)];
        const size_t expected =
            (static_cast<size_t>(block) * 4 + static_cast<size_t>(position % 4)) *
            kv_stride;
        check(table.element_offset(0, position, kv_stride) == expected,
              "offset for position " + std::to_string(position));
    }

    check(throws([&] { table.element_offset(0, 12, kv_stride); }),
          "an unbacked position throws");
    check(throws([&] { table.element_offset(5, 0, kv_stride); }),
          "an out-of-range slot throws");
    table.release_all();
}

// The trap this whole test file exists for: if every sequence happens to own one
// contiguous run of blocks, a completely wrong translation still passes. Freeing
// an interleaved slot forces slot 2's row to be non-monotonic, and the offsets
// must follow the row rather than the position.
void test_table_fragmentation() {
    pocket::QwenBlockPool pool(6, 4);
    pocket::QwenBlockTable table(&pool, 3, 32);

    check(table.ensure_capacity(0, 8), "slot 0 takes two blocks");
    check(table.ensure_capacity(1, 8), "slot 1 takes two blocks");
    const std::vector<int> freed = table.row(1);
    table.release(1);
    check(table.ensure_capacity(2, 16), "slot 2 takes the recycled blocks");

    const std::vector<int>& row = table.row(2);
    check(row.size() == 4, "slot 2 holds four blocks");
    bool contiguous = true;
    for (size_t index = 1; index < row.size(); ++index) {
        if (row[index] != row[index - 1] + 1) contiguous = false;
    }
    check(!contiguous,
          "slot 2's blocks are non-contiguous, so the case is real");

    std::set<int> distinct(row.begin(), row.end());
    check(distinct.size() == row.size(), "no block is handed out twice");
    for (const int block : freed) {
        check(distinct.count(block) == 1, "a freed block is reused");
    }

    const size_t kv_stride = 4;
    for (int position = 0; position < 16; ++position) {
        const int block = row[static_cast<size_t>(position / 4)];
        const size_t expected =
            (static_cast<size_t>(block) * 4 + static_cast<size_t>(position % 4)) *
            kv_stride;
        check(table.element_offset(2, position, kv_stride) == expected,
              "fragmented offset for position " + std::to_string(position));
    }

    // Slot 0 must be undisturbed by its neighbours' churn.
    check(table.capacity_tokens(0) == 8, "slot 0 kept its blocks");
    table.release_all();
    check(pool.free_blocks() == 6, "release_all drains every row");
}

void test_table_backpressure() {
    pocket::QwenBlockPool pool(2, 4);
    pocket::QwenBlockTable table(&pool, 2, 64);

    check(table.ensure_capacity(0, 8), "slot 0 drains the pool");
    check(pool.free_blocks() == 0, "the pool is drained");
    // Exhaustion is backpressure, not an error: the scheduler leaves the request
    // waiting, so this returns false rather than throwing.
    check(!table.ensure_capacity(1, 4), "growth fails when the pool is empty");
    check(table.capacity_tokens(1) == 0, "a failed growth backs nothing");

    table.release(0);
    check(table.ensure_capacity(1, 8), "slot 1 succeeds once blocks return");
    check(throws([&] { table.ensure_capacity(1, 65); }),
          "exceeding max_context throws rather than reporting backpressure");
    table.release_all();
}

void test_device_image() {
    pocket::QwenBlockPool pool(8, 4);
    pocket::QwenBlockTable table(&pool, 2, 16);
    check(table.dirty(), "a fresh table needs its first upload");

    check(table.ensure_capacity(0, 8), "slot 0 backed");
    const std::vector<int32_t>& image = table.device_image();
    check(image.size() == 2 * 4, "the image is max_slots x max_blocks_per_seq");
    check(!table.dirty(), "building the image clears the dirty flag");

    const std::vector<int>& row = table.row(0);
    check(image[0] == row[0] && image[1] == row[1], "row 0 is mirrored");
    // Unbacked entries must be kInvalidBlock, not zero: block 0 is a real block,
    // so zero padding would alias a live block into every short row.
    check(image[2] == pocket::QwenBlockPool::kInvalidBlock,
          "unbacked entries are kInvalidBlock");
    check(image[4] == pocket::QwenBlockPool::kInvalidBlock,
          "an empty slot's row is all kInvalidBlock");

    // A decode step that crosses no boundary must not force an upload.
    check(table.ensure_capacity(0, 8), "no growth needed");
    check(!table.dirty(), "a no-op growth leaves the image clean");
    check(table.ensure_capacity(0, 9), "crossing a boundary grows");
    check(table.dirty(), "growth marks the image dirty");
    table.release_all();
}

}  // namespace

int main() {
    test_pool_basics();
    test_pool_exhaustion();
    test_pool_refcounts();
    test_table_growth();
    test_table_translation();
    test_table_fragmentation();
    test_table_backpressure();
    test_device_image();

    if (failures != 0) {
        std::printf("%d block pool check(s) failed\n", failures);
        return 1;
    }
    std::printf("All block pool and block table checks passed\n");
    return 0;
}
