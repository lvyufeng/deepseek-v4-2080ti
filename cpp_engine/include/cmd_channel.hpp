#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace pocket {

// CPU-only command channel between TP rank 0 and worker ranks. Implemented as
// a unix domain socket bound by rank 0 and connected by ranks 1..world-1.
//
// We use this instead of NCCL broadcasts for the idle wait path so worker
// ranks block in a read() syscall (true OS sleep, 0% GPU utilization). NCCL
// is still used for prefill/decode collectives — those are short-lived and
// only run when there's real work.
class CmdChannel {
public:
    // tp_world == 1 returns an empty channel where send/recv are no-ops.
    // path_prefix is typically the NCCL id path; the socket lives at
    // "${path_prefix}.cmdsock".
    static std::unique_ptr<CmdChannel> create(int tp_world, int tp_rank,
                                              const std::string& path_prefix);

    ~CmdChannel();
    CmdChannel(const CmdChannel&) = delete;
    CmdChannel& operator=(const CmdChannel&) = delete;

    // Rank 0: write `count` int32s to every worker. Throws on I/O failure.
    void send_to_workers(const int32_t* buf, std::size_t count);

    // Rank > 0: blocking read of `count` int32s from rank 0. Throws on EOF /
    // I/O failure.
    void recv_from_root(int32_t* buf, std::size_t count);

private:
    CmdChannel();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pocket
