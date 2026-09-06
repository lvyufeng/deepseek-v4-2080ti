#include "qwen_block_table.hpp"

#include <stdexcept>
#include <string>

namespace pocket {

QwenBlockTable::QwenBlockTable(QwenBlockPool* pool, int max_slots,
                              int max_context)
    : pool_(pool), max_slots_(max_slots), max_context_(max_context) {
    if (pool_ == nullptr) {
        throw std::runtime_error("QwenBlockTable: pool must not be null");
    }
    if (max_slots <= 0) {
        throw std::runtime_error(
            "QwenBlockTable: max_slots must be >= 1, got " +
            std::to_string(max_slots));
    }
    if (max_context <= 0) {
        throw std::runtime_error(
            "QwenBlockTable: max_context must be >= 1, got " +
            std::to_string(max_context));
    }
    max_blocks_per_seq_ = pool_->blocks_for_tokens(max_context);
    rows_.assign(static_cast<size_t>(max_slots), {});
    image_.assign(static_cast<size_t>(max_slots) *
                      static_cast<size_t>(max_blocks_per_seq_),
                  QwenBlockPool::kInvalidBlock);
}

bool QwenBlockTable::ensure_capacity(int slot, int tokens) {
    validate_slot(slot);
    if (tokens < 0) {
        throw std::runtime_error(
            "QwenBlockTable::ensure_capacity: tokens must be >= 0, got " +
            std::to_string(tokens));
    }
    if (tokens > max_context_) {
        throw std::runtime_error(
            "QwenBlockTable::ensure_capacity: " + std::to_string(tokens) +
            " tokens exceeds max_context " + std::to_string(max_context_));
    }
    std::vector<int>& row = rows_[static_cast<size_t>(slot)];
    const int needed = pool_->blocks_for_tokens(tokens);
    const int have = static_cast<int>(row.size());
    if (needed <= have) return true;

    // All-or-nothing: a sequence backed to fewer positions than it will write
    // cannot run, so partial growth would only have to be undone.
    std::vector<int> fresh = pool_->allocate(needed - have);
    if (fresh.empty()) return false;

    row.insert(row.end(), fresh.begin(), fresh.end());
    dirty_ = true;
    return true;
}

void QwenBlockTable::release(int slot) {
    validate_slot(slot);
    std::vector<int>& row = rows_[static_cast<size_t>(slot)];
    if (row.empty()) return;
    pool_->free(row);
    row.clear();
    dirty_ = true;
}

void QwenBlockTable::release_all() {
    for (int slot = 0; slot < max_slots_; ++slot) release(slot);
}

size_t QwenBlockTable::element_offset(int slot, int position,
                                     size_t kv_stride) const {
    validate_slot(slot);
    if (position < 0) {
        throw std::runtime_error(
            "QwenBlockTable::element_offset: negative position " +
            std::to_string(position));
    }
    const int block_size = pool_->block_size();
    const size_t index = static_cast<size_t>(position) /
                         static_cast<size_t>(block_size);
    const std::vector<int>& row = rows_[static_cast<size_t>(slot)];
    if (index >= row.size()) {
        throw std::runtime_error(
            "QwenBlockTable::element_offset: position " +
            std::to_string(position) + " is not backed for slot " +
            std::to_string(slot));
    }
    const size_t within = static_cast<size_t>(position) %
                          static_cast<size_t>(block_size);
    return (static_cast<size_t>(row[index]) *
                static_cast<size_t>(block_size) +
            within) * kv_stride;
}

const std::vector<int>& QwenBlockTable::row(int slot) const {
    validate_slot(slot);
    return rows_[static_cast<size_t>(slot)];
}

int QwenBlockTable::capacity_tokens(int slot) const {
    validate_slot(slot);
    return static_cast<int>(rows_[static_cast<size_t>(slot)].size()) *
           pool_->block_size();
}

const std::vector<int32_t>& QwenBlockTable::device_image() {
    if (!dirty_) return image_;
    image_.assign(image_.size(), QwenBlockPool::kInvalidBlock);
    for (int slot = 0; slot < max_slots_; ++slot) {
        const std::vector<int>& row = rows_[static_cast<size_t>(slot)];
        const size_t base = static_cast<size_t>(slot) *
                            static_cast<size_t>(max_blocks_per_seq_);
        for (size_t index = 0; index < row.size(); ++index) {
            image_[base + index] = static_cast<int32_t>(row[index]);
        }
    }
    dirty_ = false;
    return image_;
}

void QwenBlockTable::validate_slot(int slot) const {
    if (slot < 0 || slot >= max_slots_) {
        throw std::runtime_error(
            "QwenBlockTable: slot " + std::to_string(slot) +
            " out of range [0, " + std::to_string(max_slots_) + ")");
    }
}

}  // namespace pocket
