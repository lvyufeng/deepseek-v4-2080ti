#pragma once

// Shared plumbing for the aclnn-backed operators.
//
// Every aclnn op is two calls: aclnn<Op>GetWorkspaceSize builds an executor and
// reports how much scratch it needs, then aclnn<Op> runs it. The workspace has to
// outlive the second call, and the executor is consumed by it. That pattern is
// identical for every op, so it lives here rather than in each file.

#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <vector>

namespace pocket {
namespace ascend {

// Scratch for one aclnn call, drawn from a per-stream pool.
//
// A matmul at Qwen's widths asks for 1.4 MB and the lm_head slice for 40 MB, and
// the graph issues thousands of ops per token, so aclrtMalloc/aclrtFree per call
// would dominate. The pool keeps one buffer per (device, stream) and grows it
// monotonically, which is safe because two ops on the same stream are serialized
// by definition: the second cannot start until the first has retired, so they
// can never hold the scratch at the same time.
//
// Growing needs care. The old buffer may still be referenced by work already
// queued on the stream, so the stream is drained before the pointer is replaced.
// Different streams never share a buffer, which is what keeps that drain local.
class WorkspacePool {
public:
    // Which buffer to draw from. A composed operator holds an intermediate across
    // several aclnn calls (the SwiGLU projections, the rstd aclnnRmsNorm insists
    // on), so that cannot come from the same buffer those calls use as their
    // workspace. Two independent slots per stream, and Intermediate is never held
    // across two Intermediate requests within one operator.
    enum class Purpose {
        OpWorkspace,
        Intermediate,
    };

    // Returns nullptr both when `bytes` is zero (aclnn accepts a null workspace
    // for ops that need no scratch) and on allocation failure; `ok` separates
    // the two.
    static void* acquire(uint64_t bytes, aclrtStream stream, bool& ok,
                         Purpose purpose = Purpose::OpWorkspace) {
        ok = true;
        if (bytes == 0) return nullptr;
        WorkspacePool& pool = instance();
        int32_t device = 0;
        if (aclrtGetDevice(&device) != ACL_SUCCESS) {
            ok = false;
            return nullptr;
        }
        std::lock_guard<std::mutex> guard(pool.mutex_);
        Slot& slot = pool.slots_[Key{device, stream, purpose}];
        if (slot.bytes >= bytes) return slot.ptr;
        if (slot.ptr != nullptr) {
            // Queued work may still be reading the buffer we are about to free.
            if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
                ok = false;
                return nullptr;
            }
            aclrtFree(slot.ptr);
            slot.ptr = nullptr;
            slot.bytes = 0;
        }
        void* ptr = nullptr;
        if (aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ok = false;
            return nullptr;
        }
        slot.ptr = ptr;
        slot.bytes = bytes;
        return ptr;
    }

private:
    struct Key {
        int32_t device;
        aclrtStream stream;
        Purpose purpose;
        bool operator<(const Key& other) const {
            if (device != other.device) return device < other.device;
            if (stream != other.stream) return stream < other.stream;
            return purpose < other.purpose;
        }
    };
    struct Slot {
        void* ptr = nullptr;
        uint64_t bytes = 0;
    };

    static WorkspacePool& instance() {
        static WorkspacePool pool;
        return pool;
    }

    std::mutex mutex_;
    std::map<Key, Slot> slots_;
};

// Scratch for one aclnn call. Borrowed from the pool, so the destructor returns
// nothing: the buffer stays alive for the next op on the same stream.
class Workspace {
public:
    Workspace(uint64_t bytes, aclrtStream stream) {
        bool ok = true;
        ptr_ = WorkspacePool::acquire(bytes, stream, ok);
        failed_ = !ok;
    }
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    void* get() const { return ptr_; }
    bool failed() const { return failed_; }

private:
    void* ptr_ = nullptr;
    bool failed_ = false;
};

// Owns the aclTensor descriptors for one op so an early return cannot leak them.
// aclDestroyTensor only releases the descriptor; the device memory it points at
// belongs to the caller.
class TensorBag {
public:
    ~TensorBag() {
        // aclDestroyTensorList destroys the members as well, and add_list has
        // already dropped them from tensors_, so these two loops do not overlap.
        for (aclTensorList* list : lists_) aclDestroyTensorList(list);
        for (aclIntArray* array : int_arrays_) aclDestroyIntArray(array);
        for (aclScalar* scalar : scalars_) aclDestroyScalar(scalar);
        for (aclTensor* tensor : tensors_) aclDestroyTensor(tensor);
    }
    TensorBag(const TensorBag&) = delete;
    TensorBag& operator=(const TensorBag&) = delete;
    TensorBag() = default;

    // Contiguous row-major tensor over `data`. Strides are derived from `shape`,
    // which is what every buffer in this engine is; a non-contiguous view needs
    // aclCreateTensor called directly.
    aclTensor* add(void* data, const std::vector<int64_t>& shape,
                   aclDataType dtype) {
        if (data == nullptr) return nullptr;
        shapes_.push_back(shape);
        std::vector<int64_t>& dims = shapes_.back();
        strides_.emplace_back(dims.size(), 1);
        std::vector<int64_t>& stride = strides_.back();
        for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
            stride[static_cast<size_t>(i)] =
                stride[static_cast<size_t>(i) + 1] * dims[static_cast<size_t>(i) + 1];
        }
        aclTensor* tensor = aclCreateTensor(
            dims.data(), static_cast<uint64_t>(dims.size()), dtype, stride.data(), 0,
            ACL_FORMAT_ND, dims.data(), static_cast<uint64_t>(dims.size()), data);
        if (tensor != nullptr) tensors_.push_back(tensor);
        return tensor;
    }

    aclTensor* add(const void* data, const std::vector<int64_t>& shape,
                   aclDataType dtype) {
        return add(const_cast<void*>(data), shape, dtype);
    }

    // Explicit strides, for a view that is not densely packed: a row-padded
    // destination, or the transposed view of a weight stored as
    // [output_rows, columns] that aclnnMatmul needs as [columns, output_rows].
    //
    // A padded stride is accepted on every operand of aclnnMatmul, including the
    // first input, provided the storage shape below covers the span the view
    // addresses. An earlier revision understated that storage and got
    // ACLNN_ERR_INNER_NULLPTR (561103) for a padded input, which read as an
    // operator limitation but was this bug.
    aclTensor* add_strided(void* data, const std::vector<int64_t>& shape,
                          const std::vector<int64_t>& stride,
                          aclDataType dtype) {
        if (data == nullptr || shape.size() != stride.size()) return nullptr;
        shapes_.push_back(shape);
        strides_.push_back(stride);
        std::vector<int64_t>& dims = shapes_.back();
        std::vector<int64_t>& view = strides_.back();

        // The last two aclCreateTensor arguments describe the *storage* behind the
        // view, not the view itself. Passing `dims` there is only correct when the
        // view is dense: with a padded or transposed stride the view addresses a
        // wider span, and a storage shape that understates it makes aclnn size its
        // internal copy from the smaller number and walk off the end. That reads
        // as a segfault inside libnnopbase rather than as an error code, so it is
        // worth computing honestly.
        //
        // The span is 1 + sum_i (dims[i] - 1) * stride[i] elements, expressed as a
        // 1-D storage shape since the layout is already carried by the strides.
        int64_t span = 1;
        for (size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] <= 0 || view[i] < 0) return nullptr;
            span += (dims[i] - 1) * view[i];
        }
        shapes_.push_back({span});
        std::vector<int64_t>& storage = shapes_.back();

        aclTensor* tensor = aclCreateTensor(
            dims.data(), static_cast<uint64_t>(dims.size()), dtype, view.data(), 0,
            ACL_FORMAT_ND, storage.data(), static_cast<uint64_t>(storage.size()),
            data);
        if (tensor != nullptr) tensors_.push_back(tensor);
        return tensor;
    }

    aclTensor* add_strided(const void* data, const std::vector<int64_t>& shape,
                          const std::vector<int64_t>& stride,
                          aclDataType dtype) {
        return add_strided(const_cast<void*>(data), shape, stride, dtype);
    }

    // A tensor list, for aclnnCat and aclnnSplitWithSize.
    //
    // aclDestroyTensorList takes ownership of the members and destroys them too
    // (measured on CANN 9.0.0: destroying the list and then the members segfaults
    // inside the *next* aclnn call, not at the second free). So a tensor handed to
    // a list is dropped from tensors_ and left to the list.
    aclTensorList* add_list(const std::vector<aclTensor*>& members) {
        for (aclTensor* member : members) {
            if (member == nullptr) return nullptr;
        }
        // Keep the pointer array alive too. The current libnnopbase copies it,
        // but aclCreateTensorList's public contract does not promise that and an
        // aclnn executor can retain the list until its second-stage call.
        lists_source_.push_back(members);
        aclTensorList* list = aclCreateTensorList(
            lists_source_.back().data(),
            static_cast<uint64_t>(lists_source_.back().size()));
        if (list == nullptr) return nullptr;
        lists_.push_back(list);
        for (aclTensor* member : members) {
            for (auto it = tensors_.begin(); it != tensors_.end(); ++it) {
                if (*it == member) {
                    tensors_.erase(it);
                    break;
                }
            }
        }
        return list;
    }

    aclIntArray* add_int_array(const std::vector<int64_t>& values) {
        arrays_source_.push_back(values);
        aclIntArray* array = aclCreateIntArray(
            arrays_source_.back().data(),
            static_cast<uint64_t>(arrays_source_.back().size()));
        if (array != nullptr) int_arrays_.push_back(array);
        return array;
    }

    aclScalar* add_scalar(float value, aclDataType dtype) {
        scalars_source_.push_back(value);
        aclScalar* scalar = aclCreateScalar(&scalars_source_.back(), dtype);
        if (scalar != nullptr) scalars_.push_back(scalar);
        return scalar;
    }

private:
    std::vector<aclTensor*> tensors_;
    std::vector<aclTensorList*> lists_;
    std::deque<std::vector<aclTensor*>> lists_source_;
    std::vector<aclIntArray*> int_arrays_;
    std::vector<aclScalar*> scalars_;
    // Deques for the same reason as the shape arrays below: aclCreateIntArray and
    // aclCreateScalar are handed pointers into these.
    std::deque<std::vector<int64_t>> arrays_source_;
    std::deque<float> scalars_source_;
    // std::deque, not std::vector: aclCreateTensor stores the shape and stride
    // pointers it is given rather than copying, so those arrays must keep their
    // addresses as later tensors are added. A vector would reallocate and leave
    // aclnn holding dangling pointers.
    std::deque<std::vector<int64_t>> shapes_;
    std::deque<std::vector<int64_t>> strides_;
};

// Resolve the stream an operator should run on. nullptr means "default stream" in
// this engine's operator contract, and unlike HCCL the aclrt* and aclnn entry
// points accept a null stream directly, so there is nothing to substitute.
inline aclrtStream resolve(void* stream) {
    return static_cast<aclrtStream>(stream);
}

// Operators return bool. Fold an aclnnStatus into that, so call sites stay flat.
inline bool ok(aclnnStatus status) { return status == ACL_SUCCESS; }

}  // namespace ascend
}  // namespace pocket
