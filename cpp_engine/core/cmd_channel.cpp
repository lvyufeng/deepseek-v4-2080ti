#include "cmd_channel.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace pocket {

namespace {

ssize_t write_all(int fd, const void* buf, std::size_t n) {
    const char* p = static_cast<const char*>(buf);
    std::size_t left = n;
    while (left > 0) {
        ssize_t w = ::write(fd, p, left);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (w == 0) return -1;
        p += w;
        left -= static_cast<std::size_t>(w);
    }
    return static_cast<ssize_t>(n);
}

ssize_t read_all(int fd, void* buf, std::size_t n) {
    char* p = static_cast<char*>(buf);
    std::size_t left = n;
    while (left > 0) {
        ssize_t r = ::read(fd, p, left);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (r == 0) return -1;
        p += r;
        left -= static_cast<std::size_t>(r);
    }
    return static_cast<ssize_t>(n);
}

}  // namespace

struct CmdChannel::Impl {
    int tp_world = 1;
    int tp_rank = 0;
    int listen_fd = -1;
    std::vector<int> peer_fds;  // rank 0: tp_world-1 fds indexed by worker rank-1; rank>0: 1 fd
    std::string sock_path;      // rank 0 only; cleared on unlink
};

CmdChannel::CmdChannel() : impl_(std::make_unique<Impl>()) {}

CmdChannel::~CmdChannel() {
    if (!impl_) return;
    for (int fd : impl_->peer_fds) if (fd >= 0) ::close(fd);
    if (impl_->listen_fd >= 0) ::close(impl_->listen_fd);
    if (impl_->tp_rank == 0 && !impl_->sock_path.empty()) {
        ::unlink(impl_->sock_path.c_str());
    }
}

std::unique_ptr<CmdChannel> CmdChannel::create(int tp_world, int tp_rank,
                                               const std::string& path_prefix) {
    std::unique_ptr<CmdChannel> cc(new CmdChannel());
    cc->impl_->tp_world = tp_world;
    cc->impl_->tp_rank = tp_rank;
    if (tp_world <= 1) return cc;
    if (path_prefix.empty()) {
        throw std::runtime_error("CmdChannel::create requires a non-empty path_prefix");
    }

    const std::string path = path_prefix + ".cmdsock";
    if (path.size() >= sizeof(sockaddr_un::sun_path)) {
        throw std::runtime_error("CmdChannel: socket path too long: " + path);
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    if (tp_rank == 0) {
        ::unlink(path.c_str());
        int lfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (lfd < 0) throw std::runtime_error(std::string("CmdChannel: socket() failed: ") + std::strerror(errno));
        if (::bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            ::close(lfd);
            throw std::runtime_error(std::string("CmdChannel: bind failed: ") + std::strerror(errno));
        }
        if (::listen(lfd, tp_world) != 0) {
            ::close(lfd);
            throw std::runtime_error(std::string("CmdChannel: listen failed: ") + std::strerror(errno));
        }
        cc->impl_->listen_fd = lfd;
        cc->impl_->sock_path = path;
        cc->impl_->peer_fds.assign(static_cast<std::size_t>(tp_world - 1), -1);
        for (int i = 0; i < tp_world - 1; ++i) {
            int cfd = ::accept(lfd, nullptr, nullptr);
            if (cfd < 0) {
                throw std::runtime_error(std::string("CmdChannel: accept failed: ") + std::strerror(errno));
            }
            int32_t their_rank = 0;
            if (read_all(cfd, &their_rank, sizeof(their_rank)) < 0) {
                ::close(cfd);
                throw std::runtime_error("CmdChannel: failed to read rank from worker");
            }
            if (their_rank < 1 || their_rank >= tp_world) {
                ::close(cfd);
                throw std::runtime_error("CmdChannel: invalid worker rank " + std::to_string(their_rank));
            }
            if (cc->impl_->peer_fds[their_rank - 1] != -1) {
                ::close(cfd);
                throw std::runtime_error("CmdChannel: duplicate worker rank " + std::to_string(their_rank));
            }
            cc->impl_->peer_fds[their_rank - 1] = cfd;
        }
    } else {
        // Worker: rank 0 may not have bound yet (we are launched concurrently
        // via mpirun/torchrun-equivalent). Retry connect for up to ~10s.
        int fd = -1;
        for (int attempt = 0; attempt < 200; ++attempt) {
            fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
            if (fd < 0) throw std::runtime_error(std::string("CmdChannel: socket() failed: ") + std::strerror(errno));
            if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) break;
            ::close(fd);
            fd = -1;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (fd < 0) {
            throw std::runtime_error("CmdChannel: connect failed after retries: " + path);
        }
        int32_t my_rank = static_cast<int32_t>(tp_rank);
        if (write_all(fd, &my_rank, sizeof(my_rank)) < 0) {
            ::close(fd);
            throw std::runtime_error("CmdChannel: failed to send rank to root");
        }
        cc->impl_->peer_fds.push_back(fd);
    }
    return cc;
}

void CmdChannel::send_to_workers(const int32_t* buf, std::size_t count) {
    if (impl_->tp_world <= 1) return;
    if (impl_->tp_rank != 0) {
        throw std::runtime_error("CmdChannel::send_to_workers called on non-root");
    }
    const std::size_t bytes = count * sizeof(int32_t);
    for (int fd : impl_->peer_fds) {
        if (write_all(fd, buf, bytes) < 0) {
            throw std::runtime_error(std::string("CmdChannel: send_to_workers write_all failed: ") + std::strerror(errno));
        }
    }
}

void CmdChannel::recv_from_root(int32_t* buf, std::size_t count) {
    if (impl_->tp_world <= 1) return;
    if (impl_->tp_rank == 0) {
        throw std::runtime_error("CmdChannel::recv_from_root called on root");
    }
    const std::size_t bytes = count * sizeof(int32_t);
    if (read_all(impl_->peer_fds.front(), buf, bytes) < 0) {
        throw std::runtime_error(std::string("CmdChannel: recv_from_root read_all failed: ") + std::strerror(errno));
    }
}

}  // namespace pocket
