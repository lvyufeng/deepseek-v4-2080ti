#include "block_pool.hpp"

#include <stdexcept>
#include <string>

namespace pocket {

BlockPool::BlockPool(int num_blocks, int block_size)
    : num_blocks_(num_blocks), block_size_(block_size) {
    if (num_blocks <= 0) {
        throw std::runtime_error(
            "BlockPool: num_blocks must be >= 1, got " +
            std::to_string(num_blocks));
    }
    if (block_size <= 0) {
        throw std::runtime_error(
            "BlockPool: block_size must be >= 1, got " +
            std::to_string(block_size));
    }
    refcounts_.assign(static_cast<size_t>(num_blocks), 0);
    free_list_.reserve(static_cast<size_t>(num_blocks));
    // Seeded in descending order so the first allocations come back as 0, 1, 2,
    // which keeps a fresh pool's block tables readable in a debugger and makes
    // the fragmentation test's interleaving reproducible.
    for (int block = num_blocks - 1; block >= 0; --block) {
        free_list_.push_back(block);
    }
}

std::vector<int> BlockPool::allocate(int count) {
    if (count < 0) {
        throw std::runtime_error(
            "BlockPool::allocate: count must be >= 0, got " +
            std::to_string(count));
    }
    std::vector<int> blocks;
    if (count == 0) return blocks;
    if (static_cast<size_t>(count) > free_list_.size()) return blocks;

    blocks.reserve(static_cast<size_t>(count));
    for (int taken = 0; taken < count; ++taken) {
        const int block = free_list_.back();
        free_list_.pop_back();
        refcounts_[static_cast<size_t>(block)] = 1;
        blocks.push_back(block);
    }
    return blocks;
}

void BlockPool::free(const std::vector<int>& block_ids) {
    for (const int block : block_ids) {
        if (block == kInvalidBlock) continue;
        validate(block);
        int& count = refcounts_[static_cast<size_t>(block)];
        if (count == 0) {
            throw std::runtime_error(
                "BlockPool::free: block " + std::to_string(block) +
                " is already free");
        }
        if (--count == 0) {
            free_list_.push_back(block);
        }
    }
}

void BlockPool::retain(int block_id) {
    validate(block_id);
    int& count = refcounts_[static_cast<size_t>(block_id)];
    if (count == 0) {
        throw std::runtime_error(
            "BlockPool::retain: block " + std::to_string(block_id) +
            " is not allocated");
    }
    ++count;
}

int BlockPool::refcount(int block_id) const {
    validate(block_id);
    return refcounts_[static_cast<size_t>(block_id)];
}

int BlockPool::blocks_for_tokens(int tokens) const {
    if (tokens < 0) {
        throw std::runtime_error(
            "BlockPool::blocks_for_tokens: tokens must be >= 0, got " +
            std::to_string(tokens));
    }
    return (tokens + block_size_ - 1) / block_size_;
}

void BlockPool::validate(int block_id) const {
    if (block_id < 0 || block_id >= num_blocks_) {
        throw std::runtime_error(
            "BlockPool: block id " + std::to_string(block_id) +
            " out of range [0, " + std::to_string(num_blocks_) + ")");
    }
}

}  // namespace pocket
