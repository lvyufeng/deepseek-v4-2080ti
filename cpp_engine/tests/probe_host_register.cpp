// Whole-GGUF cudaHostRegister is BANNED; this probe is a no-op guard.
//
// Registering a file-backed 80GB+ GGUF mmap pins file pages and poisons
// Linux dirty-page accounting / writeback (balance_dirty_pages), stalling
// all unrelated fsync/git/build I/O system-wide. This binary used to
// measure that register + the async H2D bandwidth it enabled, but exercising
// it is itself the hazard, so it no longer touches cudaHostRegister at all.
// Expert H2D goes through pageable copies or a bounded anonymous pinned
// staging ring instead. This file remains only as a regression guard: if a
// future change reintroduces whole-mmap pinning, the accompanying grep-based
// CI check (no cudaHostRegister in cpp_engine sources) should fail.

#include "gguf_reader.hpp"

#include <cstdio>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: probe_host_register <model.gguf>\n";
        return 2;
    }
    // Open the file (read-only mmap) but never cudaHostRegister it.
    pocket::GGUFFile gguf(argv[1]);
    const size_t bytes = gguf.file_size();
    std::printf("gguf size = %.2f GB\n", bytes / (1024.0 * 1024.0 * 1024.0));
    std::printf("whole-GGUF cudaHostRegister is BANNED; probe is a no-op guard.\n");
    return 0;
}
