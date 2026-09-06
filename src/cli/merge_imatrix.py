from __future__ import annotations

from argparse import ArgumentParser

from src.loader.gguf.imatrix import merge_imatrix_files


def parse_args():
    parser = ArgumentParser(description="Merge rank-sharded routed MoE imatrix GGUF files.")
    parser.add_argument("--output", required=True, help="Merged imatrix GGUF path")
    parser.add_argument("--dataset", default="pocketllm_runtime_capture_merged")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override imatrix.chunk_size; defaults to first input")
    parser.add_argument("inputs", nargs="+", help="Rank-local imatrix GGUF inputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_imatrix_files(args.inputs, args.output, dataset=args.dataset, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
