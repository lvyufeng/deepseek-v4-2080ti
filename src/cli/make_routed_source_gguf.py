from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np

from src.loader.gguf.reader import GGUF_MAGIC, GGUFReader, GGML_TYPES, align_up, tensor_nbytes


FP4_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

F16_GGML_TYPE = 1

_METADATA_SCALAR_SIZES = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}


@dataclass(frozen=True)
class SafeTensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_begin: int
    data_end: int
    absolute_begin: int
    nbytes: int


class SafeTensorShard:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = os.path.abspath(os.fspath(path))
        self._fd = os.open(self.path, os.O_RDONLY)
        self._mmap = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
        self._data_start = 0
        self.tensors: dict[str, SafeTensorInfo] = {}
        self._parse_header()

    def close(self) -> None:
        mapped = getattr(self, "_mmap", None)
        if mapped is not None:
            try:
                mapped.close()
            except BufferError:
                # NumPy views created with frombuffer() can outlive the immediate
                # call site until the generator frame is collected.  The process
                # is about to exit in CLI use; keep the mmap alive rather than
                # masking the original exception with a close-time BufferError.
                pass
            else:
                self._mmap = None
        fd = getattr(self, "_fd", -1)
        if fd >= 0:
            os.close(fd)
            self._fd = -1

    def __enter__(self) -> "SafeTensorShard":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _parse_header(self) -> None:
        if len(self._mmap) < 8:
            raise ValueError(f"safetensors shard too small: {self.path}")
        header_len = struct.unpack_from("<Q", self._mmap, 0)[0]
        self._data_start = 8 + int(header_len)
        if self._data_start > len(self._mmap):
            raise ValueError(f"safetensors header exceeds file size: {self.path}")
        header = json.loads(self._mmap[8:self._data_start].decode("utf-8"))
        for name, record in header.items():
            if name == "__metadata__":
                continue
            offsets = record["data_offsets"]
            if len(offsets) != 2 or offsets[1] < offsets[0]:
                raise ValueError(f"bad data_offsets for {name} in {self.path}")
            shape = tuple(int(v) for v in record["shape"])
            begin = int(offsets[0])
            end = int(offsets[1])
            self.tensors[name] = SafeTensorInfo(
                name=name,
                dtype=str(record["dtype"]),
                shape=shape,
                data_begin=begin,
                data_end=end,
                absolute_begin=self._data_start + begin,
                nbytes=end - begin,
            )

    def info(self, name: str) -> SafeTensorInfo:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"safetensors tensor not found in {self.path}: {name}") from exc

    def array_u8(self, info: SafeTensorInfo) -> np.ndarray:
        return np.frombuffer(self._mmap, dtype=np.uint8, count=info.nbytes, offset=info.absolute_begin)


class SafeTensorIndex:
    def __init__(self, ckpt_dir: str | os.PathLike[str]):
        self.ckpt_dir = os.path.abspath(os.fspath(ckpt_dir))
        index_path = os.path.join(self.ckpt_dir, "model.safetensors.index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        self.weight_map: dict[str, str] = dict(index["weight_map"])
        self._shards: dict[str, SafeTensorShard] = {}

    def close(self) -> None:
        for shard in self._shards.values():
            shard.close()
        self._shards.clear()

    def __enter__(self) -> "SafeTensorIndex":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def shard_for(self, tensor_name: str) -> SafeTensorShard:
        try:
            shard_name = self.weight_map[tensor_name]
        except KeyError as exc:
            raise KeyError(f"tensor missing from safetensors index: {tensor_name}") from exc
        shard = self._shards.get(shard_name)
        if shard is None:
            shard = SafeTensorShard(os.path.join(self.ckpt_dir, shard_name))
            self._shards[shard_name] = shard
        return shard

    def tensor_pair(self, weight_name: str) -> tuple[SafeTensorShard, SafeTensorInfo, SafeTensorInfo]:
        scale_name = weight_name[:-7] + ".scale" if weight_name.endswith(".weight") else weight_name + ".scale"
        weight_shard_name = self.weight_map.get(weight_name)
        scale_shard_name = self.weight_map.get(scale_name)
        if weight_shard_name is None:
            raise KeyError(f"FP4 weight not found: {weight_name}")
        if scale_shard_name is None:
            raise KeyError(f"FP4 scale not found: {scale_name}")
        if weight_shard_name != scale_shard_name:
            raise ValueError(f"FP4 weight and scale are in different shards: {weight_name}")
        shard = self.shard_for(weight_name)
        return shard, shard.info(weight_name), shard.info(scale_name)


@dataclass(frozen=True)
class RoutedTensorSpec:
    layer: int
    role: str
    source_suffix: str


@dataclass(frozen=True)
class RewritePlan:
    type_ids: dict[str, int]
    nbytes: dict[str, int]
    total_size: int
    converted_routed_tensors: int
    converted_routed_bytes: int


def _read_struct(f: BinaryIO, fmt: str):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise EOFError("unexpected end of file")
    values = struct.unpack(fmt, data)
    return values[0] if len(values) == 1 else values


def _read_string(f: BinaryIO) -> str:
    n = _read_struct(f, "<Q")
    data = f.read(n)
    if len(data) != n:
        raise EOFError("unexpected end of string")
    return data.decode("utf-8", errors="replace")


def _skip_string(f: BinaryIO) -> None:
    n = _read_struct(f, "<Q")
    f.seek(n, os.SEEK_CUR)


def _skip_metadata_value(f: BinaryIO, value_type: int) -> None:
    if value_type == 8:  # string
        _skip_string(f)
        return
    if value_type == 9:  # array
        item_type = _read_struct(f, "<I")
        length = _read_struct(f, "<Q")
        if item_type == 8:
            for _ in range(int(length)):
                _skip_string(f)
            return
        if item_type == 9:
            raise ValueError("nested GGUF metadata arrays are not supported")
        try:
            item_size = _METADATA_SCALAR_SIZES[item_type]
        except KeyError as exc:
            raise ValueError(f"unsupported GGUF array item type {item_type}") from exc
        f.seek(int(length) * item_size, os.SEEK_CUR)
        return
    try:
        size = _METADATA_SCALAR_SIZES[value_type]
    except KeyError as exc:
        raise ValueError(f"unsupported GGUF metadata type {value_type}") from exc
    f.seek(size, os.SEEK_CUR)


def _metadata_raw_bytes(path: str, metadata_count: int) -> bytes:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        f.seek(4 + 8 + 8, os.SEEK_CUR)  # version + tensor_count + metadata_count
        meta_start = f.tell()
        for _ in range(metadata_count):
            _skip_string(f)
            value_type = _read_struct(f, "<I")
            _skip_metadata_value(f, value_type)
        meta_end = f.tell()
        f.seek(meta_start)
        return f.read(meta_end - meta_start)


def _write_string(f: BinaryIO, value: str) -> None:
    data = value.encode("utf-8")
    f.write(struct.pack("<Q", len(data)))
    f.write(data)


def _write_tensor_records(f: BinaryIO, tensors, type_ids: dict[str, int], offsets: dict[str, int]) -> None:
    for tensor in tensors:
        _write_string(f, tensor.name)
        f.write(struct.pack("<I", len(tensor.dimensions)))
        for dim in tensor.dimensions:
            f.write(struct.pack("<Q", int(dim)))
        f.write(struct.pack("<I", int(type_ids[tensor.name])))
        f.write(struct.pack("<Q", int(offsets[tensor.name])))


def _pad_file(f: BinaryIO, alignment: int) -> None:
    pos = f.tell()
    padded = align_up(pos, alignment)
    if padded > pos:
        f.write(b"\0" * (padded - pos))


def _copy_bytes(src_fd: int, dst: BinaryIO, offset: int, nbytes: int, chunk_bytes: int) -> None:
    remaining = int(nbytes)
    pos = int(offset)
    while remaining:
        size = min(remaining, chunk_bytes)
        data = os.pread(src_fd, size, pos)
        if len(data) != size:
            raise EOFError(f"short read while copying template data at {pos}: got {len(data)}, expected {size}")
        dst.write(data)
        pos += size
        remaining -= size


def _routed_spec(name: str) -> RoutedTensorSpec | None:
    parts = name.split(".")
    if len(parts) != 4 or parts[0] != "blk" or parts[3] != "weight":
        return None
    try:
        layer = int(parts[1])
    except ValueError:
        return None
    role = parts[2]
    if role == "ffn_gate_exps":
        return RoutedTensorSpec(layer=layer, role=role, source_suffix="w1")
    if role == "ffn_down_exps":
        return RoutedTensorSpec(layer=layer, role=role, source_suffix="w2")
    if role == "ffn_up_exps":
        return RoutedTensorSpec(layer=layer, role=role, source_suffix="w3")
    return None


def _parse_layers(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == "":
        return None
    layers: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo_s, hi_s = item.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                raise ValueError(f"bad layer range: {item}")
            layers.update(range(lo, hi + 1))
        else:
            layers.add(int(item))
    return layers


def _selected_routed_tensor(name: str, layers: set[int] | None, roles: set[str] | None, max_count: int | None, selected_so_far: int) -> bool:
    spec = _routed_spec(name)
    if spec is None:
        return False
    if layers is not None and spec.layer not in layers:
        return False
    if roles is not None and spec.role not in roles:
        return False
    if max_count is not None and selected_so_far >= max_count:
        return False
    return True


def _plan_rewrite(ds4, layers: set[int] | None, roles: set[str] | None, max_routed_tensors: int | None) -> RewritePlan:
    type_ids: dict[str, int] = {}
    nbytes_map: dict[str, int] = {}
    total_size = 0
    selected = 0
    selected_bytes = 0
    for tensor in ds4.tensors:
        if _selected_routed_tensor(tensor.name, layers, roles, max_routed_tensors, selected):
            nbytes = tensor.elements * 2
            type_id = F16_GGML_TYPE
            selected += 1
            selected_bytes += nbytes
        else:
            if tensor.nbytes is None:
                raise ValueError(f"cannot determine nbytes for tensor {tensor.name} type_id={tensor.type_id}")
            nbytes = int(tensor.nbytes)
            type_id = int(tensor.type_id)
        type_ids[tensor.name] = type_id
        nbytes_map[tensor.name] = nbytes
        total_size += align_up(nbytes, int(ds4.alignment))
    return RewritePlan(type_ids, nbytes_map, total_size, selected, selected_bytes)


def _source_weight_name(spec: RoutedTensorSpec, expert: int) -> str:
    return f"layers.{spec.layer}.ffn.experts.{expert}.{spec.source_suffix}.weight"


def _iter_dequant_fp4_f16_chunks(
    weight_u8: np.ndarray,
    scale_u8: np.ndarray,
    rows: int,
    in_dim: int,
    row_chunk: int,
    row_limit: int | None = None,
) -> Iterable[np.ndarray]:
    packed_cols = in_dim // 2
    scale_cols = in_dim // 32
    total_rows = int(weight_u8.size // packed_cols)
    if int(scale_u8.size // scale_cols) != total_rows:
        raise ValueError("FP4 weight/scale row count mismatch")
    if rows > total_rows:
        raise ValueError(f"requested {rows} rows but source only has {total_rows}")
    weight_u8 = weight_u8.reshape(total_rows, packed_cols)
    scale_u8 = scale_u8.reshape(total_rows, scale_cols)
    rows_to_emit = rows if row_limit is None else min(rows, int(row_limit))
    row_chunk = max(1, int(row_chunk))
    for row_start in range(0, rows_to_emit, row_chunk):
        row_end = min(rows, row_start + row_chunk)
        raw = weight_u8[row_start:row_end]
        out = np.empty((row_end - row_start, in_dim), dtype=np.float32)
        out[:, 0::2] = FP4_TABLE[raw & 0x0F]
        out[:, 1::2] = FP4_TABLE[raw >> 4]
        exponents = scale_u8[row_start:row_end].astype(np.int16) - 127
        scales = np.ldexp(np.ones_like(exponents, dtype=np.float32), exponents).repeat(32, axis=1)
        out *= scales[:, :in_dim]
        yield out.astype("<f2", copy=False)


def _write_routed_tensor_f16(
    dst: BinaryIO,
    safe_index: SafeTensorIndex,
    tensor_name: str,
    dimensions: tuple[int, ...],
    row_chunk: int,
) -> None:
    spec = _routed_spec(tensor_name)
    if spec is None:
        raise ValueError(f"not a routed tensor: {tensor_name}")
    if len(dimensions) != 3:
        raise ValueError(f"routed GGUF tensor must be 3D: {tensor_name} dims={dimensions}")
    in_dim, out_dim, n_experts = (int(v) for v in dimensions)
    if in_dim % 32 != 0:
        raise ValueError(f"routed tensor input dim must be divisible by 32: {tensor_name} in_dim={in_dim}")
    written = 0
    for expert in range(n_experts):
        source_name = _source_weight_name(spec, expert)
        shard, weight_info, scale_info = safe_index.tensor_pair(source_name)
        expected_weight_shape = (out_dim, in_dim // 2)
        expected_scale_shape = (out_dim, in_dim // 32)
        if weight_info.dtype != "I8" or tuple(weight_info.shape) != expected_weight_shape:
            raise ValueError(
                f"source FP4 weight shape/type mismatch for {source_name}: "
                f"dtype={weight_info.dtype} shape={weight_info.shape}, expected I8 {expected_weight_shape}"
            )
        if scale_info.dtype != "F8_E8M0" or tuple(scale_info.shape) != expected_scale_shape:
            raise ValueError(
                f"source FP4 scale shape/type mismatch for {source_name}: "
                f"dtype={scale_info.dtype} shape={scale_info.shape}, expected F8_E8M0 {expected_scale_shape}"
            )
        weight_u8 = shard.array_u8(weight_info)
        scale_u8 = shard.array_u8(scale_info)
        for chunk in _iter_dequant_fp4_f16_chunks(weight_u8, scale_u8, out_dim, in_dim, row_chunk):
            dst.write(chunk)
            written += chunk.nbytes
    expected = math.prod(int(v) for v in dimensions) * 2
    if written != expected:
        raise RuntimeError(f"wrong byte count for {tensor_name}: wrote {written}, expected {expected}")


def _format_size(nbytes: int) -> str:
    return f"{nbytes / 1024 ** 3:.2f} GiB ({nbytes / 1024 ** 2:.0f} MiB)"


def _smoke_first_selected(ds4, safe_index: SafeTensorIndex, layers: set[int] | None, roles: set[str] | None) -> None:
    for tensor in ds4.tensors:
        if not _selected_routed_tensor(tensor.name, layers, roles, 1, 0):
            continue
        spec = _routed_spec(tensor.name)
        assert spec is not None
        in_dim, out_dim, _n_experts = (int(v) for v in tensor.dimensions)
        source_name = _source_weight_name(spec, 0)
        shard, weight_info, scale_info = safe_index.tensor_pair(source_name)
        weight_u8 = shard.array_u8(weight_info)
        scale_u8 = shard.array_u8(scale_info)
        chunk = next(_iter_dequant_fp4_f16_chunks(weight_u8, scale_u8, out_dim, in_dim, row_chunk=8, row_limit=8))
        finite = bool(np.isfinite(chunk.astype(np.float32)).all())
        print(
            f"smoke source={source_name} gguf={tensor.name} chunk_shape={chunk.shape} "
            f"dtype={chunk.dtype} min={float(chunk.min()):.6g} max={float(chunk.max()):.6g} finite={finite}",
            flush=True,
        )
        return
    print("smoke: no routed tensor selected", flush=True)


def rewrite_gguf(args: argparse.Namespace) -> None:
    template = os.path.abspath(args.template_gguf)
    output = os.path.abspath(args.output)
    layers = _parse_layers(args.layers)
    roles = set(args.roles.split(",")) if args.roles else None
    if roles is not None:
        allowed = {"ffn_gate_exps", "ffn_down_exps", "ffn_up_exps"}
        bad = roles - allowed
        if bad:
            raise ValueError(f"unknown routed roles: {sorted(bad)}")

    ds4 = GGUFReader(template).read()
    plan = _plan_rewrite(ds4, layers, roles, args.max_routed_tensors)
    original_payload = sum(align_up(int(t.nbytes or 0), int(ds4.alignment)) for t in ds4.tensors)
    print(f"template tensors={ds4.tensor_count} alignment={ds4.alignment} data_start={ds4.data_start}", flush=True)
    print(f"selected routed tensors -> F16: {plan.converted_routed_tensors}", flush=True)
    print(f"original padded payload: {_format_size(original_payload)}", flush=True)
    print(f"planned padded payload : {_format_size(plan.total_size)}", flush=True)
    if plan.converted_routed_tensors:
        print(f"converted routed raw bytes: {_format_size(plan.converted_routed_bytes)}", flush=True)

    with SafeTensorIndex(args.hf_ckpt_path) as safe_index:
        if args.smoke_first_expert:
            _smoke_first_selected(ds4, safe_index, layers, roles)
        if args.dry_run:
            return

        if os.path.exists(output):
            raise FileExistsError(f"output already exists; refusing to overwrite: {output}")
        tmp_output = output + ".tmp"
        if os.path.exists(tmp_output):
            raise FileExistsError(f"temporary output already exists; rename or remove it first: {tmp_output}")

        metadata_raw = _metadata_raw_bytes(template, ds4.metadata_count)
        offsets: dict[str, int] = {}
        off = 0
        for tensor in ds4.tensors:
            offsets[tensor.name] = off
            off += align_up(plan.nbytes[tensor.name], int(ds4.alignment))

        src_fd = os.open(template, os.O_RDONLY)
        try:
            converted = 0
            with open(tmp_output, "xb") as out:
                out.write(GGUF_MAGIC)
                out.write(struct.pack("<I", int(ds4.version)))
                out.write(struct.pack("<Q", int(ds4.tensor_count)))
                out.write(struct.pack("<Q", int(ds4.metadata_count)))
                out.write(metadata_raw)
                _write_tensor_records(out, ds4.tensors, plan.type_ids, offsets)
                _pad_file(out, int(ds4.alignment))
                data_start = out.tell()
                if data_start != ds4.data_start:
                    print(f"warning: output data_start={data_start} differs from template data_start={ds4.data_start}", flush=True)
                for idx, tensor in enumerate(ds4.tensors, 1):
                    if idx == 1 or idx == ds4.tensor_count or idx % 25 == 0:
                        print(f"write tensor {idx}/{ds4.tensor_count}: {tensor.name}", flush=True)
                    if plan.type_ids[tensor.name] == F16_GGML_TYPE and _routed_spec(tensor.name) is not None:
                        _write_routed_tensor_f16(out, safe_index, tensor.name, tensor.dimensions, args.row_chunk)
                        converted += 1
                    else:
                        if tensor.nbytes is None:
                            raise ValueError(f"cannot copy unknown-size tensor: {tensor.name}")
                        _copy_bytes(src_fd, out, tensor.absolute_offset, int(tensor.nbytes), args.copy_chunk_bytes)
                    _pad_file(out, int(ds4.alignment))
            os.replace(tmp_output, output)
            print(f"wrote {output}", flush=True)
            print(f"converted routed tensors: {converted}", flush=True)
        finally:
            os.close(src_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a quantization-source GGUF for DeepSeek-V4 by preserving template metadata/non-routed tensors "
            "and replacing selected routed FP4 safetensors experts with F16 GGUF tensors."
        )
    )
    parser.add_argument("--template-gguf", required=True, help="Existing DeepSeek-V4 GGUF template with tokenizer/metadata/non-routed tensors")
    parser.add_argument("--hf-ckpt-path", required=True, help="Original DeepSeek-V4-Flash FP4 safetensors directory")
    parser.add_argument("--output", required=True, help="Output GGUF path; must not already exist")
    parser.add_argument("--layers", default=None, help="Optional comma/range list, e.g. '0,3-5'. Default: all layers")
    parser.add_argument(
        "--roles",
        default=None,
        help="Optional comma list from ffn_gate_exps,ffn_down_exps,ffn_up_exps. Default: all routed roles",
    )
    parser.add_argument("--max-routed-tensors", type=int, default=None, help="Debug limit for converted routed tensors; remaining tensors are copied")
    parser.add_argument("--row-chunk", type=int, default=64, help="Rows per FP4->F16 dequant chunk")
    parser.add_argument("--copy-chunk-bytes", type=int, default=64 * 1024 * 1024, help="Copy chunk size for preserved template tensors")
    parser.add_argument("--dry-run", action="store_true", help="Print rewrite plan without writing output")
    parser.add_argument("--smoke-first-expert", action="store_true", help="Dequantize a small slice from the first selected routed expert")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rewrite_gguf(args)


if __name__ == "__main__":
    main()
