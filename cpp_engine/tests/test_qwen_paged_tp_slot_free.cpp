// TP worker block-leak regression: freeing a paged slot on rank 0 must also
// release that slot's blocks on every worker rank.
//
// The paged block pool lives per rank, and a slot's table row and KV blocks
// live on whatever rank runs the forward. So when rank 0's scheduler calls
// free_slot() it broadcasts a FreeSlot command; each worker releases its own
// copy of the slot's blocks. Before that command existed, workers kept a slot's
// blocks for the life of the process: rank 0's pool returned to full after each
// free while the workers' pools drained one slot per free, until a worker ran
// out at some later prefill -- asynchronously, and only under TP.
//
// The single-process tests cannot see this (there is no worker to leak), and
// the TP paged admission bench does not free slots. So the only way to pin the
// leak is a TP run that repeatedly allocate -> prefill -> decode -> free and
// forces the worker pool small enough that a worker-side leak exhausts it
// within the loop.
//
// Engineering the pool to be that tight is the whole trick. The fixed pool
// keeps enough blocks for the working set, so a correct engine passes every
// cycle; a leaky worker crosses the pool the moment its cumulative leaked
// blocks exceed the budget, and the failure surfaces as the next command's
// reserve_paged_slot() throwing inside run_worker_loop().

#include "qwen_engine.hpp"
#include "qwen_parity_fixture.hpp"
#include "cuda_ops.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct Options {
    int layers = 2;
    int max_context = 128;
    int tp_world = 1;
    int tp_rank = 0;
    int device = -1;
    std::string nccl_id_path;
};

void expect(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "[ok]  " : "[FAIL]", what.c_str());
}

// Number of alloc->prefill->decode->free cycles. The working set per cycle is
// a single 16-token block (2-token prompt + one decode step never crosses a
// block boundary), so with the fix the pool sees a constant 1-block demand and
// never approaches the budget. A worker that retains one slot's blocks per
// freed slot leaks that 1 block every cycle and must run the pool dry inside
// this loop.
constexpr int kCycles = 24;

// The fixture has kv_heads=4, head_dim=128, one full-attention layer (layer 0
// is linear_attention and carries the qkv weight, layer 1 has none), so
//   bytes_per_block = block_size * local_kv_heads * head_dim * K,V * full_layers
// Under TP2 each rank owns 2 KV heads: 16 * 2 * 128 * 2 * 1 = 16384 B/block.
constexpr int kBlockSize = 16;
constexpr int kBlocksPerSeq =
    (128 + kBlockSize - 1) / kBlockSize;  // 8 blocks to reach max_context
// A pool of 16 blocks (262144 B) sits 2x above the one-max-context floor, so a
// worker that retains one slot's blocks per freed slot leaks ~1-2 blocks every
// cycle and exhausts the pool inside this loop.
constexpr uint64_t kPoolBytes = 16 * 16384;

static_assert(kPoolBytes >= kBlocksPerSeq * 16384ULL,
              "pool must hold a lone max-context sequence under TP2");

}  // namespace

int main(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--layers" && i + 1 < argc) {
            opts.layers = std::atoi(argv[++i]);
        } else if (arg == "--max-context" && i + 1 < argc) {
            opts.max_context = std::atoi(argv[++i]);
        } else if (arg == "--tp-world" && i + 1 < argc) {
            opts.tp_world = std::atoi(argv[++i]);
        } else if (arg == "--tp-rank" && i + 1 < argc) {
            opts.tp_rank = std::atoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            opts.device = std::atoi(argv[++i]);
        } else if (arg == "--nccl-id" && i + 1 < argc) {
            opts.nccl_id_path = argv[++i];
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }

    if (!dsv4::cuda_runtime_available()) {
        std::printf("[SKIP] test_qwen_paged_tp_slot_free requires CUDA\n");
        return 0;
    }

    try {
        // Fixture path is deterministic per name; build it once on rank 0
        // before the engine is constructed, so the worker ranks read the same
        // directory without racing rank 0's write. warmup_tp() synchronises all
        // ranks afterwards, but it runs after this write (see below).
        const std::string checkpoint =
            qwen_fixture::fixture_dir("qwen_paged_engine_parity_fixture");
        if (opts.tp_rank == 0) {
            // The parity fixture's FP8 MLP (128 columns) does not shard onto the
            // FP8 scale block at TP2, so this regression uses the BF16-MLP
            // variant. Geometry is identical; the KV-pool behaviour under test
            // is unaffected by the MLP dtype.
            if (!qwen_fixture::write_fixture(checkpoint, /*bf16_mlp=*/true)) {
                throw std::runtime_error("could not create paged parity fixture");
            }
        } else {
            // Worker ranks start while rank 0 is still writing. wait for the
            // two files the engine opens at construction; rank 0's write is
            // small and completes in milliseconds, so a bounded poll is enough
            // and avoids a startup race that is otherwise flaky.
            const std::string shard = checkpoint + "/model.safetensors";
            const std::string config = checkpoint + "/config.json";
            for (int attempt = 0; attempt < 2000; ++attempt) {
                if (access(shard.c_str(), F_OK) == 0 &&
                    access(config.c_str(), F_OK) == 0) {
                    break;
                }
                usleep(1000);
            }
        }

        dsv4::QwenEngineOptions engine_opts;
        engine_opts.device = opts.device >= 0 ? opts.device : opts.tp_rank;
        engine_opts.max_batch_size = 2;
        engine_opts.kv_cache_dtype = dsv4::QwenKvCacheDType::Fp16;
        engine_opts.prefix_cache = false;
        engine_opts.kv_paged = true;
        engine_opts.kv_block_size = kBlockSize;
        engine_opts.kv_cache_bytes = kPoolBytes;
        if (opts.tp_world > 1) {
            engine_opts.tp_world = opts.tp_world;
            engine_opts.tp_rank = opts.tp_rank;
            engine_opts.nccl_id_path = opts.nccl_id_path;
        }

        dsv4::QwenEngine engine(checkpoint, engine_opts, opts.layers,
                                opts.max_context);
        engine.allocate_batch_slots(2);
        engine.warmup_tp();

        // Workers spend the rest of their life in the command loop, then check
        // their OWN pool. This is the assertion that matters: rank 0 cannot read
        // a worker's pool over any channel, so the leak is only visible from
        // inside the worker process. run_worker_loop returns on Shutdown, by
        // which point rank 0 has freed every slot it allocated; a worker that
        // honoured each FreeSlot is back to a full pool, and a worker that
        // ignored them is short one row per freed slot.
        if (opts.tp_rank != 0) {
            engine.run_worker_loop();
            const int free_blocks = engine.kv_free_blocks();
            const int total_blocks = engine.kv_total_blocks();
            std::printf("rank %d after shutdown: %d/%d blocks free\n",
                        opts.tp_rank, free_blocks, total_blocks);
            if (free_blocks != total_blocks) {
                std::printf(
                    "[FAIL] rank %d leaked %d block(s): FreeSlot did not "
                    "release worker-side paged state\n",
                    opts.tp_rank, total_blocks - free_blocks);
                return 1;
            }
            return 0;
        }

        std::printf("\nTP paged slot-free regression\n");
        std::printf("=============================\n");
        std::printf("TP %d, pool %d blocks of %d, %d cycles\n\n", opts.tp_world,
                    engine.kv_total_blocks(), kBlockSize, kCycles);

        const int total = engine.kv_total_blocks();

        // A 2-token prompt plus one decode step needs a single block at any
        // point: the sequence stays inside one 16-token block, so the working
        // set is one block and the pool floor is what dominates.
        const std::vector<int> prompt = {1, 2};
        dsv4::QwenBatchSamplingParams sampling;
        sampling.max_new_tokens = kCycles;  // never the stopping bound
        sampling.ignore_eos = true;

        for (int cycle = 0; cycle < kCycles; ++cycle) {
            const uint64_t request_id = 1000 + static_cast<uint64_t>(cycle);
            const int slot = engine.allocate_slot(request_id);
            if (slot < 0) {
                expect(false, "slot could not be allocated");
                engine.worker_command_shutdown();
                return 1;
            }

            auto req = std::make_unique<dsv4::QwenBatchedRequest>();
            req->request_id = request_id;
            req->prompt_tokens = prompt;
            req->slot_id = slot;
            req->sampling = sampling;
            std::vector<dsv4::QwenBatchedRequest*> batch{req.get()};

            const dsv4::QwenBatchPrefillResult prefilled =
                engine.batch_prefill(batch, 0);
            if (prefilled.results.empty()) {
                expect(false, "prefill returned no result");
                engine.worker_command_shutdown();
                return 1;
            }
            engine.batch_decode_step(batch);

            // Rank 0's pool must return to full here. This catches a rank-0
            // block leak directly. A worker-only leak is caught by the loop
            // failing below instead -- that is the regression this test exists
            // to guard.
            engine.free_slot(request_id);
            if (engine.kv_free_blocks() != total) {
                expect(false, "slot free returned pool to full");
                engine.worker_command_shutdown();
                return 1;
            }
        }

        expect(true, "pool never exhausted across cycles");
        expect(engine.kv_free_blocks() == total,
               "pool is back to its total after every free");

        engine.worker_command_shutdown();

        std::printf("\nPASS\n");
        return 0;
    } catch (const std::exception& ex) {
        std::printf("[FAIL] test_qwen_paged_tp_slot_free %s\n", ex.what());
        return 1;
    }
}
