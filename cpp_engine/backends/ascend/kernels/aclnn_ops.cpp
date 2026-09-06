// Ascend implementations of the Qwen operators that map onto aclnn library ops.
//
// Scope: matrix products, normalizations, elementwise activations, and the
// reshape-shaped operators (concat, split, strided copy, region gather/scatter).
// The recurrent and attention kernels are not here; they have no aclnn equivalent
// on this install and live in hand-written AscendC.
//
// Three conventions that are easy to get wrong, all measured on CANN 9.0.0 with
// first-generation 910 (Short_SoC_version=Ascend910):
//
//   1. Weights are stored row-major as [output_rows, columns] and the product is
//      y[b, r] = dot(x[b, :], weight[r, :]). aclnnMatmul computes a plain
//      A @ B, so B is passed as a transposed *view* of the same memory --
//      shape [columns, output_rows], stride {1, weight_stride} -- rather than
//      being physically transposed. This is verified working.
//   2. A strided view must declare a storage shape covering the span it actually
//      addresses, not just its own dims. Understating it does not produce an error
//      at the boundary: aclnn either returns ACLNN_ERR_INNER_NULLPTR (561103) or
//      corrupts the heap and segfaults inside a *later* call. TensorBag::add_strided
//      computes the span, so padded strides work on every operand.
//   3. Qwen RMSNorm is (1 + weight) but aclnnRmsNorm applies gamma directly, so
//      the +1 is folded into the FP16 gamma at upload time by
//      qwen_apply_norm_gamma_policy(). aclnnRmsNorm also rejects an FP32 gamma
//      against FP16 activations (161002), so the fold has to happen in FP16.
//
// A related ownership rule, for the same reason: aclDestroyTensorList destroys its
// members, so a tensor handed to a list must not also be destroyed individually.
// TensorBag handles this; see add_list.
//
// aclnnAddRmsNorm is not usable here: it is rejected outright on this SoC
// (561103), so the fused residual-add-norm operator is composed by hand.

#include "aclnn_common.hpp"

#include "device_runtime.hpp"
#include "qwen_ascend_ops.hpp"

#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_cat.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_embedding.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_split_with_size.h>

#include <cstring>
#include <vector>

namespace pocket {
namespace {

using ascend::ok;
using ascend::resolve;
using ascend::TensorBag;
using ascend::Workspace;

// Device scratch for an intermediate the caller did not provide: the rstd output
// aclnnRmsNorm insists on, the two SwiGLU projections before the pointwise pass.
//
// This comes from the pool's Intermediate slot rather than from device_malloc, for
// the same reason the aclnn workspace does: the graph issues thousands of ops per
// token and a malloc/free pair per call would dominate. It is a distinct slot from
// the workspace because an intermediate stays live across the aclnn calls that read
// it. No operator here holds two intermediates at once; a future one that needs to
// must take a single larger buffer and sub-divide it, as the SwiGLU path does.
class Scratch {
public:
    Scratch(size_t bytes, aclrtStream stream) {
        bool ok = true;
        ptr_ = ascend::WorkspacePool::acquire(
            bytes, stream, ok, ascend::WorkspacePool::Purpose::Intermediate);
        failed_ = !ok;
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;

    void* get() const { return ptr_; }
    bool failed() const { return failed_; }

private:
    void* ptr_ = nullptr;
    bool failed_ = false;
};

// Every aclnn op is the same two-phase shape: size a workspace from the built
// executor, then run it. `plan` fills in workspace size and executor; `run`
// consumes both. Wrapping the pair keeps each operator down to its argument
// marshalling.
template <typename Plan, typename Run>
bool invoke(Plan plan, Run run, aclrtStream stream) {
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    if (!ok(plan(&bytes, &executor)) || executor == nullptr) return false;
    Workspace workspace(bytes, stream);
    if (workspace.failed()) return false;
    return ok(run(workspace.get(), bytes, executor));
}

// y[b, r] = dot(x[b, :], weight[r, :]) for either an FP16 or an FP32 output.
bool matmul_rows(const uint16_t* x, const uint16_t* weight, void* y,
                 aclDataType out_dtype, int batch, int rows, int cols,
                 int x_stride, int y_stride, int weight_stride, void* stream) {
    if (x == nullptr || weight == nullptr || y == nullptr) return false;
    if (batch <= 0 || rows <= 0 || cols <= 0) return false;
    if (x_stride < cols || y_stride < rows || weight_stride < cols) return false;

    const int64_t b = batch;
    const int64_t r = rows;
    const int64_t c = cols;
    TensorBag bag;
    aclTensor* a = bag.add_strided(x, {b, c}, {static_cast<int64_t>(x_stride), 1},
                                   ACL_FLOAT16);
    aclTensor* w = bag.add_strided(weight, {c, r},
                                   {1, static_cast<int64_t>(weight_stride)},
                                   ACL_FLOAT16);
    aclTensor* out = bag.add_strided(y, {b, r},
                                     {static_cast<int64_t>(y_stride), 1},
                                     out_dtype);
    if (a == nullptr || w == nullptr || out == nullptr) return false;

    const aclrtStream s = resolve(stream);
    return invoke(
        [&](uint64_t* bytes, aclOpExecutor** executor) {
            // cubeMathType=1 keeps FP16 inputs on the cube unit without an
            // implicit widening pass; measured max_rel_err 3.7e-04 against a
            // double-precision host reference.
            return aclnnMatmulGetWorkspaceSize(a, w, out, 1, bytes, executor);
        },
        [&](void* workspace, uint64_t bytes, aclOpExecutor* executor) {
            return aclnnMatmul(workspace, bytes, executor, s);
        },
        s);
}

// rms(x) * gamma over the last dimension, writing FP16. `rstd` is required by
// aclnnRmsNorm even when unused.
bool rms_norm(const uint16_t* x, const uint16_t* gamma, uint16_t* y, int rows,
              int cols, float eps, aclrtStream stream) {
    Scratch rstd(static_cast<size_t>(rows) * sizeof(float), stream);
    if (rstd.failed()) return false;
    TensorBag bag;
    const int64_t r = rows;
    const int64_t c = cols;
    aclTensor* in = bag.add(x, {r, c}, ACL_FLOAT16);
    aclTensor* g = bag.add(gamma, {c}, ACL_FLOAT16);
    aclTensor* out = bag.add(y, {r, c}, ACL_FLOAT16);
    aclTensor* rstd_tensor = bag.add(rstd.get(), {r, 1}, ACL_FLOAT);
    if (in == nullptr || g == nullptr || out == nullptr ||
        rstd_tensor == nullptr) {
        return false;
    }
    return invoke(
        [&](uint64_t* bytes, aclOpExecutor** executor) {
            return aclnnRmsNormGetWorkspaceSize(in, g, static_cast<double>(eps),
                                                out, rstd_tensor, bytes,
                                                executor);
        },
        [&](void* workspace, uint64_t bytes, aclOpExecutor* executor) {
            return aclnnRmsNorm(workspace, bytes, executor, stream);
        },
        stream);
}

// dst *= src, elementwise over `count` FP16 values.
bool multiply_inplace(uint16_t* dst, const uint16_t* src, int64_t count,
                      aclrtStream stream) {
    TensorBag bag;
    aclTensor* target = bag.add(dst, {count}, ACL_FLOAT16);
    aclTensor* other = bag.add(src, {count}, ACL_FLOAT16);
    if (target == nullptr || other == nullptr) return false;
    return invoke(
        [&](uint64_t* bytes, aclOpExecutor** executor) {
            return aclnnInplaceMulGetWorkspaceSize(target, other, bytes,
                                                   executor);
        },
        [&](void* workspace, uint64_t bytes, aclOpExecutor* executor) {
            return aclnnInplaceMul(workspace, bytes, executor, stream);
        },
        stream);
}

// out = silu(in), elementwise over `count` FP16 values.
bool silu(const uint16_t* in, uint16_t* out, int64_t count, aclrtStream stream) {
    TensorBag bag;
    aclTensor* self = bag.add(in, {count}, ACL_FLOAT16);
    aclTensor* result = bag.add(out, {count}, ACL_FLOAT16);
    if (self == nullptr || result == nullptr) return false;
    return invoke(
        [&](uint64_t* bytes, aclOpExecutor** executor) {
            return aclnnSiluGetWorkspaceSize(self, result, bytes, executor);
        },
        [&](void* workspace, uint64_t bytes, aclOpExecutor* executor) {
            return aclnnSilu(workspace, bytes, executor, stream);
        },
        stream);
}

// Copy `src` into `dst`, honouring both views' strides. This is the general
// reshape primitive behind the split and strided-copy operators.
bool copy_view(aclTensor* dst, aclTensor* src, aclrtStream stream) {
    if (dst == nullptr || src == nullptr) return false;
    return invoke(
        [&](uint64_t* bytes, aclOpExecutor** executor) {
            return aclnnInplaceCopyGetWorkspaceSize(dst, src, bytes, executor);
        },
        [&](void* workspace, uint64_t bytes, aclOpExecutor* executor) {
            return aclnnInplaceCopy(workspace, bytes, executor, stream);
        },
        stream);
}

}  // namespace

bool qwen_fp16_matmul_rows_f16_ascend(const uint16_t* d_x_fp16,
                                      const uint16_t* d_w_fp16,
                                      uint16_t* d_y_fp16, int batch, int rows,
                                      int cols, int x_stride, int y_stride,
                                      int weight_stride, void* stream) {
    return matmul_rows(d_x_fp16, d_w_fp16, d_y_fp16, ACL_FLOAT16, batch, rows,
                       cols, x_stride, y_stride, weight_stride, stream);
}

bool qwen_fp16_matmul_rows_f16_f32_ascend(const uint16_t* d_x_fp16,
                                          const uint16_t* d_w_fp16, float* d_y,
                                          int batch, int rows, int cols,
                                          int x_stride, int y_stride,
                                          int weight_stride, void* stream) {
    return matmul_rows(d_x_fp16, d_w_fp16, d_y, ACL_FLOAT, batch, rows, cols,
                       x_stride, y_stride, weight_stride, stream);
}

// Gate and up projections followed by silu(gate) * up. There is no fused aclnn
// form, so this is the two matmuls plus the pointwise pass; the CUDA kernel fuses
// them, which is one of the hotspots earmarked for AscendC.
bool qwen_fp16_swiglu_matmul_rows_f16_ascend(
    const uint16_t* d_x_fp16, const uint16_t* d_gate_fp16,
    const uint16_t* d_up_fp16, uint16_t* d_y_fp16, int batch, int rows, int cols,
    int x_stride, int y_stride, int weight_stride, void* stream) {
    if (d_y_fp16 == nullptr || batch <= 0 || rows <= 0 || y_stride < rows) {
        return false;
    }
    // Both projections land in densely packed scratch rather than one of them
    // going straight to d_y_fp16. The reason is aclnnSilu: there is no
    // aclnnInplaceSilu on this install, and nothing documents whether passing the
    // same buffer as self and out is legal, so the SiLU reads scratch and writes
    // d_y_fp16 with no aliasing to rely on. The cost is one extra batch*rows FP16
    // buffer, which is small next to the two matmul workspaces.
    const size_t elements = static_cast<size_t>(batch) * static_cast<size_t>(rows);
    Scratch projections(2 * elements * sizeof(uint16_t), resolve(stream));
    if (projections.failed()) return false;
    auto* gate_rows = static_cast<uint16_t*>(projections.get());
    uint16_t* up_rows = gate_rows + elements;

    if (!matmul_rows(d_x_fp16, d_gate_fp16, gate_rows, ACL_FLOAT16, batch, rows,
                     cols, x_stride, rows, weight_stride, stream)) {
        return false;
    }
    if (!matmul_rows(d_x_fp16, d_up_fp16, up_rows, ACL_FLOAT16, batch, rows, cols,
                     x_stride, rows, weight_stride, stream)) {
        return false;
    }
    // y_stride may exceed rows, so the pointwise pass walks one row at a time
    // rather than treating the output as a flat run.
    const aclrtStream s = resolve(stream);
    for (int b = 0; b < batch; ++b) {
        uint16_t* y_row = d_y_fp16 + static_cast<size_t>(b) * y_stride;
        const uint16_t* gate_row = gate_rows + static_cast<size_t>(b) * rows;
        const uint16_t* up_row = up_rows + static_cast<size_t>(b) * rows;
        if (!silu(gate_row, y_row, rows, s)) return false;
        if (!multiply_inplace(y_row, up_row, rows, s)) return false;
    }
    return true;
}

bool qwen_rmsnorm_fp16_gamma_rows_f16_ascend(const uint16_t* d_x_fp16,
                                             const uint16_t* d_gamma_fp16,
                                             uint16_t* d_y_fp16, int rows,
                                             int cols, float eps, void* stream) {
    if (d_x_fp16 == nullptr || d_gamma_fp16 == nullptr || d_y_fp16 == nullptr) {
        return false;
    }
    if (rows <= 0 || cols <= 0 || eps < 0.0f) return false;
    return rms_norm(d_x_fp16, d_gamma_fp16, d_y_fp16, rows, cols, eps,
                    resolve(stream));
}

// residual = hidden + delta, then normalized = rms(residual) * gamma. The CUDA
// kernel fuses these into one pass and rounds the sum to FP16 before the sum of
// squares; doing the add first into `residual` reproduces that rounding exactly,
// which matters because the two are compared for token parity.
bool qwen_residual_add_rmsnorm_fp16_gamma_rows_f16_ascend(
    const uint16_t* d_hidden_fp16, const uint16_t* d_delta_fp16,
    const uint16_t* d_gamma_fp16, uint16_t* d_residual_fp16,
    uint16_t* d_normalized_fp16, int rows, int cols, float eps, void* stream) {
    if (d_hidden_fp16 == nullptr || d_delta_fp16 == nullptr ||
        d_gamma_fp16 == nullptr || d_residual_fp16 == nullptr ||
        d_normalized_fp16 == nullptr) {
        return false;
    }
    if (rows <= 0 || cols <= 0 || eps < 0.0f) return false;
    const aclrtStream s = resolve(stream);
    const size_t bytes =
        static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(uint16_t);
    // hidden and residual are the same buffer at some call sites and distinct at
    // others, so seed the output with hidden and add delta in place.
    if (d_residual_fp16 != d_hidden_fp16) {
        if (!memcpy_d2d_async(d_residual_fp16, d_hidden_fp16, bytes, stream)) {
            return false;
        }
    }
    const int64_t count =
        static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    TensorBag bag;
    aclTensor* target = bag.add(d_residual_fp16, {count}, ACL_FLOAT16);
    aclTensor* other = bag.add(d_delta_fp16, {count}, ACL_FLOAT16);
    aclScalar* alpha = bag.add_scalar(1.0f, ACL_FLOAT);
    if (target == nullptr || other == nullptr || alpha == nullptr) return false;
    const bool added = invoke(
        [&](uint64_t* size, aclOpExecutor** executor) {
            return aclnnInplaceAddGetWorkspaceSize(target, other, alpha, size,
                                                   executor);
        },
        [&](void* workspace, uint64_t size, aclOpExecutor* executor) {
            return aclnnInplaceAdd(workspace, size, executor, s);
        },
        s);
    if (!added) return false;
    return rms_norm(d_residual_fp16, d_gamma_fp16, d_normalized_fp16, rows, cols,
                    eps, s);
}

// rms(x) * gamma * silu(gate). Unlike the two norms above, this one applies gamma
// directly with no (1 + w) fold; see qwen_ascend_norm_gamma_bias.
bool qwen_gated_rmsnorm_fp16_gamma_rows_f16_ascend(
    const uint16_t* d_x_fp16, const uint16_t* d_gamma_fp16,
    const uint16_t* d_gate_fp16, uint16_t* d_y_fp16, int rows, int cols,
    float eps, void* stream) {
    if (d_x_fp16 == nullptr || d_gamma_fp16 == nullptr ||
        d_gate_fp16 == nullptr || d_y_fp16 == nullptr) {
        return false;
    }
    if (rows <= 0 || cols <= 0 || eps < 0.0f) return false;
    const aclrtStream s = resolve(stream);
    if (!rms_norm(d_x_fp16, d_gamma_fp16, d_y_fp16, rows, cols, eps, s)) {
        return false;
    }
    const int64_t count =
        static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    Scratch activated(static_cast<size_t>(count) * sizeof(uint16_t), s);
    if (activated.failed()) return false;
    auto* gate_activated = static_cast<uint16_t*>(activated.get());
    if (!silu(d_gate_fp16, gate_activated, count, s)) return false;
    return multiply_inplace(d_y_fp16, gate_activated, count, s);
}

bool qwen_add_inplace_f16_ascend(uint16_t* d_y_fp16, const uint16_t* d_x_fp16,
                                 int count, void* stream) {
    if (d_y_fp16 == nullptr || d_x_fp16 == nullptr || count <= 0) return false;
    const aclrtStream s = resolve(stream);
    TensorBag bag;
    aclTensor* target = bag.add(d_y_fp16, {count}, ACL_FLOAT16);
    aclTensor* other = bag.add(d_x_fp16, {count}, ACL_FLOAT16);
    aclScalar* alpha = bag.add_scalar(1.0f, ACL_FLOAT);
    if (target == nullptr || other == nullptr || alpha == nullptr) return false;
    return invoke(
        [&](uint64_t* bytes, aclOpExecutor** executor) {
            return aclnnInplaceAddGetWorkspaceSize(target, other, alpha, bytes,
                                                   executor);
        },
        [&](void* workspace, uint64_t bytes, aclOpExecutor* executor) {
            return aclnnInplaceAdd(workspace, bytes, executor, s);
        },
        s);
}

bool qwen_sigmoid_mul_f16_ascend(const uint16_t* d_x_fp16,
                                 const uint16_t* d_gate_fp16,
                                 uint16_t* d_y_fp16, int count, void* stream) {
    if (d_x_fp16 == nullptr || d_gate_fp16 == nullptr || d_y_fp16 == nullptr ||
        count <= 0) {
        return false;
    }
    const aclrtStream s = resolve(stream);
    const int64_t n = count;
    // Sigmoid into the output first, then multiply by x. Writing the activation
    // to d_y_fp16 avoids a second scratch buffer and is safe even when d_y_fp16
    // aliases d_x_fp16, because the multiply reads x after the write.
    Scratch activated(static_cast<size_t>(n) * sizeof(uint16_t), s);
    if (activated.failed()) return false;
    auto* gate_activated = static_cast<uint16_t*>(activated.get());
    {
        TensorBag bag;
        aclTensor* self = bag.add(d_gate_fp16, {n}, ACL_FLOAT16);
        aclTensor* out = bag.add(gate_activated, {n}, ACL_FLOAT16);
        if (self == nullptr || out == nullptr) return false;
        const bool applied = invoke(
            [&](uint64_t* bytes, aclOpExecutor** executor) {
                return aclnnSigmoidGetWorkspaceSize(self, out, bytes, executor);
            },
            [&](void* workspace, uint64_t bytes, aclOpExecutor* executor) {
                return aclnnSigmoid(workspace, bytes, executor, s);
            },
            s);
        if (!applied) return false;
    }
    if (d_y_fp16 != d_x_fp16) {
        if (!memcpy_d2d_async(d_y_fp16, d_x_fp16,
                              static_cast<size_t>(n) * sizeof(uint16_t),
                              stream)) {
            return false;
        }
    }
    return multiply_inplace(d_y_fp16, gate_activated, n, s);
}

bool qwen_silu_mul_rows_f16_ascend(const uint16_t* d_gate_fp16,
                                   const uint16_t* d_up_fp16,
                                   uint16_t* d_y_fp16, int rows, int cols,
                                   void* stream) {
    if (d_gate_fp16 == nullptr || d_up_fp16 == nullptr || d_y_fp16 == nullptr ||
        rows <= 0 || cols <= 0) {
        return false;
    }
    const aclrtStream s = resolve(stream);
    const int64_t count = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (!silu(d_gate_fp16, d_y_fp16, count, s)) return false;
    return multiply_inplace(d_y_fp16, d_up_fp16, count, s);
}

bool qwen_concat_rows_f16_ascend(const uint16_t* d_left,
                                 const uint16_t* d_right, uint16_t* d_out,
                                 int rows, int cols, void* stream) {
    if (d_left == nullptr || d_right == nullptr || d_out == nullptr ||
        rows <= 0 || cols <= 0) {
        return false;
    }
    const aclrtStream s = resolve(stream);
    const int64_t r = rows;
    const int64_t c = cols;
    TensorBag bag;
    aclTensor* left = bag.add(d_left, {r, c}, ACL_FLOAT16);
    aclTensor* right = bag.add(d_right, {r, c}, ACL_FLOAT16);
    aclTensor* out = bag.add(d_out, {r, 2 * c}, ACL_FLOAT16);
    if (left == nullptr || right == nullptr || out == nullptr) return false;
    aclTensorList* parts = bag.add_list({left, right});
    if (parts == nullptr) return false;
    return invoke(
        [&](uint64_t* bytes, aclOpExecutor** executor) {
            return aclnnCatGetWorkspaceSize(parts, 1, out, bytes, executor);
        },
        [&](void* workspace, uint64_t bytes, aclOpExecutor* executor) {
            return aclnnCat(workspace, bytes, executor, s);
        },
        s);
}

bool qwen_copy_rows_strided_f16_ascend(const uint16_t* d_source_fp16,
                                       int source_row_stride,
                                       uint16_t* d_destination_fp16,
                                       int destination_row_stride, int rows,
                                       int columns, void* stream) {
    if (d_source_fp16 == nullptr || d_destination_fp16 == nullptr || rows <= 0 ||
        columns <= 0 || source_row_stride < columns ||
        destination_row_stride < columns) {
        return false;
    }
    const aclrtStream s = resolve(stream);
    const int64_t r = rows;
    const int64_t c = columns;
    TensorBag bag;
    aclTensor* src = bag.add_strided(
        d_source_fp16, {r, c}, {static_cast<int64_t>(source_row_stride), 1},
        ACL_FLOAT16);
    aclTensor* dst = bag.add_strided(
        d_destination_fp16, {r, c},
        {static_cast<int64_t>(destination_row_stride), 1}, ACL_FLOAT16);
    return copy_view(dst, src, s);
}

// packed [rows, 2*key_dim + value_dim] -> q, k [rows, key_dim], v [rows, value_dim].
bool qwen_split_packed_qkv_f16_ascend(const uint16_t* d_packed_fp16,
                                      uint16_t* d_q_fp16, uint16_t* d_k_fp16,
                                      uint16_t* d_v_fp16, int rows, int key_dim,
                                      int value_dim, void* stream) {
    if (d_packed_fp16 == nullptr || d_q_fp16 == nullptr || d_k_fp16 == nullptr ||
        d_v_fp16 == nullptr || rows <= 0 || key_dim <= 0 || value_dim <= 0) {
        return false;
    }
    const aclrtStream s = resolve(stream);
    const int64_t r = rows;
    const int64_t k = key_dim;
    const int64_t v = value_dim;
    TensorBag bag;
    aclTensor* packed = bag.add(d_packed_fp16, {r, 2 * k + v}, ACL_FLOAT16);
    aclTensor* q = bag.add(d_q_fp16, {r, k}, ACL_FLOAT16);
    aclTensor* key = bag.add(d_k_fp16, {r, k}, ACL_FLOAT16);
    aclTensor* value = bag.add(d_v_fp16, {r, v}, ACL_FLOAT16);
    if (packed == nullptr || q == nullptr || key == nullptr ||
        value == nullptr) {
        return false;
    }
    aclTensorList* outputs = bag.add_list({q, key, value});
    aclIntArray* sizes = bag.add_int_array({k, k, v});
    if (outputs == nullptr || sizes == nullptr) return false;
    return invoke(
        [&](uint64_t* bytes, aclOpExecutor** executor) {
            return aclnnSplitWithSizeGetWorkspaceSize(packed, sizes, 1, outputs,
                                                      bytes, executor);
        },
        [&](void* workspace, uint64_t bytes, aclOpExecutor* executor) {
            return aclnnSplitWithSize(workspace, bytes, executor, s);
        },
        s);
}

// [rows, 2*width] -> two [rows, width] halves.
bool qwen_split_rows_pair_f16_ascend(const uint16_t* d_packed_fp16,
                                     uint16_t* d_first_fp16,
                                     uint16_t* d_second_fp16, int rows,
                                     int width, void* stream) {
    if (d_packed_fp16 == nullptr || d_first_fp16 == nullptr ||
        d_second_fp16 == nullptr || rows <= 0 || width <= 0) {
        return false;
    }
    const aclrtStream s = resolve(stream);
    const int64_t r = rows;
    const int64_t w = width;
    TensorBag bag;
    aclTensor* packed = bag.add(d_packed_fp16, {r, 2 * w}, ACL_FLOAT16);
    aclTensor* first = bag.add(d_first_fp16, {r, w}, ACL_FLOAT16);
    aclTensor* second = bag.add(d_second_fp16, {r, w}, ACL_FLOAT16);
    if (packed == nullptr || first == nullptr || second == nullptr) return false;
    aclTensorList* outputs = bag.add_list({first, second});
    aclIntArray* sizes = bag.add_int_array({w, w});
    if (outputs == nullptr || sizes == nullptr) return false;
    return invoke(
        [&](uint64_t* bytes, aclOpExecutor** executor) {
            return aclnnSplitWithSizeGetWorkspaceSize(packed, sizes, 1, outputs,
                                                      bytes, executor);
        },
        [&](void* workspace, uint64_t bytes, aclOpExecutor* executor) {
            return aclnnSplitWithSize(workspace, bytes, executor, s);
        },
        s);
}

// Gather embedding rows for tokens that fall inside this rank's vocab slice,
// zeroing the rest so the caller's all-reduce reassembles the full row.
//
// aclnnEmbedding does not range-check: an index past the end of the table returns
// garbage rather than faulting (measured). Under vocab-sharded TP most tokens are
// outside a given rank's slice, so that is the common path, not an edge case.
// Indices are therefore clamped into range before the gather and the rows that
// were out of range are multiplied by zero after it.
//
// The clamp and mask are computed on the host, which costs one device-to-host copy
// of `count` ints. That is a stream synchronization point, and it is the reason
// this operator is on the AscendC replacement list; it is not a correctness
// compromise. The surrounding engine code uploads the tokens synchronously
// anyway, so no asynchrony is lost today.
bool qwen_embedding_fp16_gather_f16_ascend(const uint16_t* d_table_fp16,
                                           const int* d_tokens,
                                           uint16_t* d_out_fp16, int count,
                                           int cols, int row_start,
                                           int row_count, void* stream) {
    if (d_table_fp16 == nullptr || d_tokens == nullptr || d_out_fp16 == nullptr) {
        return false;
    }
    if (count <= 0 || cols <= 0 || row_start < 0 || row_count <= 0) return false;

    std::vector<int32_t> tokens(static_cast<size_t>(count));
    if (!memcpy_d2h(tokens.data(), d_tokens,
                    static_cast<size_t>(count) * sizeof(int32_t))) {
        return false;
    }
    std::vector<int32_t> local(static_cast<size_t>(count));
    std::vector<uint16_t> mask(static_cast<size_t>(count));
    bool any_masked = false;
    for (int i = 0; i < count; ++i) {
        const int32_t token = tokens[static_cast<size_t>(i)];
        const bool resident = token >= row_start && token < row_start + row_count;
        local[static_cast<size_t>(i)] = resident ? token - row_start : 0;
        // 0x3C00 is 1.0 in IEEE FP16, 0x0000 is +0.
        mask[static_cast<size_t>(i)] = resident ? 0x3C00u : 0x0000u;
        if (!resident) any_masked = true;
    }

    // One buffer sub-divided into the clamped indices and the row mask. Two
    // separate Scratch objects would draw the same pooled slot and alias, and the
    // mask has to survive the gather that reads the indices.
    const size_t index_bytes = static_cast<size_t>(count) * sizeof(int32_t);
    const size_t mask_bytes = static_cast<size_t>(count) * sizeof(uint16_t);
    const aclrtStream s = resolve(stream);
    Scratch staging(index_bytes + mask_bytes, s);
    if (staging.failed()) return false;
    auto* index_device = static_cast<uint8_t*>(staging.get());
    uint8_t* mask_device = index_device + index_bytes;
    if (!memcpy_h2d(index_device, local.data(), index_bytes)) return false;
    if (any_masked && !memcpy_h2d(mask_device, mask.data(), mask_bytes)) {
        return false;
    }

    {
        TensorBag bag;
        aclTensor* table = bag.add(d_table_fp16,
                                   {static_cast<int64_t>(row_count),
                                    static_cast<int64_t>(cols)},
                                   ACL_FLOAT16);
        aclTensor* index = bag.add(index_device, {static_cast<int64_t>(count)},
                                   ACL_INT32);
        aclTensor* out = bag.add(d_out_fp16,
                                 {static_cast<int64_t>(count),
                                  static_cast<int64_t>(cols)},
                                 ACL_FLOAT16);
        if (table == nullptr || index == nullptr || out == nullptr) return false;
        const bool gathered = invoke(
            [&](uint64_t* bytes, aclOpExecutor** executor) {
                return aclnnEmbeddingGetWorkspaceSize(table, index, out, bytes,
                                                      executor);
            },
            [&](void* workspace, uint64_t bytes, aclOpExecutor* executor) {
                return aclnnEmbedding(workspace, bytes, executor, s);
            },
            s);
        if (!gathered) return false;
    }
    if (!any_masked) return true;

    TensorBag bag;
    // [count, 1] against [count, cols]: the broadcast is applied by aclnn, so the
    // mask stays one value per row rather than a full-width buffer.
    aclTensor* out = bag.add(d_out_fp16,
                             {static_cast<int64_t>(count),
                              static_cast<int64_t>(cols)},
                             ACL_FLOAT16);
    aclTensor* row_mask = bag.add(mask_device,
                                  {static_cast<int64_t>(count), 1}, ACL_FLOAT16);
    if (out == nullptr || row_mask == nullptr) return false;
    return invoke(
        [&](uint64_t* bytes, aclOpExecutor** executor) {
            return aclnnInplaceMulGetWorkspaceSize(out, row_mask, bytes,
                                                   executor);
        },
        [&](void* workspace, uint64_t bytes, aclOpExecutor* executor) {
            return aclnnInplaceMul(workspace, bytes, executor, s);
        },
        s);
}

namespace {

// Move independently allocated device regions to or from their offsets in one
// packed transaction buffer. The CUDA kernel copies fixed-size blocks so that one
// launch covers every region; here one asynchronous copy per region is issued
// instead, which is the same traffic without the block bookkeeping. `total_blocks`
// only sized the CUDA grid, so it is validated rather than used.
bool copy_regions(const QwenCopyRegion* regions, int region_count,
                  uint8_t* packed, uint64_t total_blocks, bool gather,
                  void* stream) {
    if (regions == nullptr || packed == nullptr || region_count <= 0 ||
        total_blocks == 0) {
        return false;
    }
    std::vector<QwenCopyRegion> table(static_cast<size_t>(region_count));
    if (!memcpy_d2h(table.data(), regions,
                    static_cast<size_t>(region_count) * sizeof(QwenCopyRegion))) {
        return false;
    }
    uint64_t counted_blocks = 0;
    for (const QwenCopyRegion& region : table) {
        if (region.device_address == 0 || region.bytes == 0) return false;
        if (region.first_block != counted_blocks) return false;
        counted_blocks += qwen_copy_region_blocks(region.bytes);
    }
    if (counted_blocks != total_blocks) return false;

    for (const QwenCopyRegion& region : table) {
        auto* region_data = reinterpret_cast<uint8_t*>(region.device_address);
        uint8_t* packed_data = packed + region.packed_offset;
        void* destination = gather ? packed_data : region_data;
        const void* source = gather ? region_data : packed_data;
        if (!memcpy_d2d_async(destination, source,
                              static_cast<size_t>(region.bytes), stream)) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool qwen_gather_copy_regions_ascend(const QwenCopyRegion* d_regions,
                                     int region_count, uint8_t* d_packed,
                                     uint64_t total_blocks, void* stream) {
    return copy_regions(d_regions, region_count, d_packed, total_blocks, true,
                        stream);
}

bool qwen_scatter_copy_regions_ascend(const QwenCopyRegion* d_regions,
                                      int region_count, const uint8_t* d_packed,
                                      uint64_t total_blocks, void* stream) {
    return copy_regions(d_regions, region_count,
                        const_cast<uint8_t*>(d_packed), total_blocks, false,
                        stream);
}

// Deinterleave [rows, q_heads, 2*head_dim] into q and gate, each
// [rows, q_heads, head_dim]. The split is on the innermost dimension, so this is
// a 3-D split rather than the 2-D form above.
bool qwen_split_q_gate_f16_ascend(const uint16_t* d_q_proj_fp16,
                                  uint16_t* d_q_fp16, uint16_t* d_gate_fp16,
                                  int rows, int q_heads, int head_dim,
                                  void* stream) {
    if (d_q_proj_fp16 == nullptr || d_q_fp16 == nullptr ||
        d_gate_fp16 == nullptr || rows <= 0 || q_heads <= 0 || head_dim <= 0) {
        return false;
    }
    const aclrtStream s = resolve(stream);
    const int64_t r = rows;
    const int64_t h = q_heads;
    const int64_t d = head_dim;
    TensorBag bag;
    aclTensor* source = bag.add(d_q_proj_fp16, {r, h, 2 * d}, ACL_FLOAT16);
    aclTensor* q = bag.add(d_q_fp16, {r, h, d}, ACL_FLOAT16);
    aclTensor* gate = bag.add(d_gate_fp16, {r, h, d}, ACL_FLOAT16);
    if (source == nullptr || q == nullptr || gate == nullptr) return false;
    aclTensorList* outputs = bag.add_list({q, gate});
    aclIntArray* sizes = bag.add_int_array({d, d});
    if (outputs == nullptr || sizes == nullptr) return false;
    return invoke(
        [&](uint64_t* bytes, aclOpExecutor** executor) {
            return aclnnSplitWithSizeGetWorkspaceSize(source, sizes, 2, outputs,
                                                      bytes, executor);
        },
        [&](void* workspace, uint64_t bytes, aclOpExecutor* executor) {
            return aclnnSplitWithSize(workspace, bytes, executor, s);
        },
        s);
}

}  // namespace pocket
