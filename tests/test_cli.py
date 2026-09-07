from __future__ import annotations

import pytest

from pocketllm.api import BackendCapabilities, UnsupportedFeatureError
from pocketllm.backends.base import BackendBase
import pocketllm.cli as cli
from pocketllm.cli import _args, _supervised_command, build_parser


class _FakeBackend(BackendBase):
    def __init__(self) -> None:
        super().__init__()
        self._ready = True
        self.prepare_calls = 0
        self.worker_calls = 0

    @property
    def capabilities(self):
        return BackendCapabilities(name="fake", supports_streaming=True)

    def generate(self, requests):
        return []

    def stream(self, request):
        yield from ()

    def prepare(self) -> None:
        self.prepare_calls += 1
        self._ready = True

    def run_worker(self, on_ready=None) -> None:
        self.worker_calls += 1
        if on_ready is not None:
            on_ready()


def _parse(*extra: str):
    return build_parser().parse_args(["serve", "--model", "checkpoint", *extra])


def test_cli_maps_common_engine_fields() -> None:
    args = _args(_parse(
        "--backend", "torch",
        "--tensor-parallel-size", "2",
        "--max-model-len", "16384",
        "--kv-cache-dtype", "fp8",
        "--prefill-chunk-tokens", "4096",
        "--no-enable-prefix-caching",
    ))

    assert args.backend == "torch"
    assert args.tensor_parallel_size == 2
    assert args.max_model_len == 16384
    assert args.kv_cache_dtype == "fp8"
    assert args.prefill_chunk_tokens == 4096
    assert args.enable_prefix_caching is False


def test_cli_exposes_attention_window_and_speculation() -> None:
    args = _args(_parse(
        "--attention-window", "4096",
        "--attention-sink-tokens", "128",
        "--speculative-method", "mtp",
        "--speculative-tokens", "3",
    ))

    assert args.attention_window == 4096
    assert args.attention_sink_tokens == 128
    assert args.speculative_method == "mtp"
    assert args.speculative_tokens == 3


def test_backend_option_values_are_json_typed() -> None:
    args = _args(_parse(
        "--backend-option", "max_state_snapshots=16",
        "--backend-option", "mtp_adaptive=true",
        "--backend-option", "dspark_checkpoint=/models/drafter",
    ))

    assert args.backend_options["max_state_snapshots"] == 16
    assert args.backend_options["mtp_adaptive"] is True
    assert args.backend_options["dspark_checkpoint"] == "/models/drafter"
    assert args.backend_options["engine_kind"] == "auto"


def test_backend_option_requires_key_value_form() -> None:
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        _args(_parse("--backend-option", "bare-flag"))


def test_tensor_parallel_supervisor_flags() -> None:
    args = _args(_parse(
        "--tensor-parallel-size", "4",
        "--tensor-parallel-startup-timeout", "600",
        "--tensor-parallel-shutdown-timeout", "60",
    ))
    assert args.tensor_parallel_size == 4
    namespace = _parse(
        "--tensor-parallel-startup-timeout", "600",
        "--tensor-parallel-shutdown-timeout", "60",
    )
    assert namespace.tensor_parallel_startup_timeout == 600.0
    assert namespace.tensor_parallel_shutdown_timeout == 60.0
    assert namespace.tensor_parallel_master_addr is None
    assert namespace.tensor_parallel_master_port is None
    assert namespace.tensor_parallel_rendezvous_dir is None
    assert namespace.tensor_parallel_supervisor is True


def test_tensor_parallel_supervisor_opt_out() -> None:
    assert _parse("--no-tensor-parallel-supervisor").tensor_parallel_supervisor is False


def test_tensor_parallel_rendezvous_options() -> None:
    namespace = _parse(
        "--tensor-parallel-master-addr", "127.0.0.1",
        "--tensor-parallel-master-port", "23456",
        "--tensor-parallel-rendezvous-dir", "/var/tmp/pocketllm",
    )
    assert namespace.tensor_parallel_master_addr == "127.0.0.1"
    assert namespace.tensor_parallel_master_port == 23456
    assert namespace.tensor_parallel_rendezvous_dir == "/var/tmp/pocketllm"


def test_served_model_name_defaults_to_model_path() -> None:
    namespace = _parse()
    assert namespace.served_model_name is None
    assert _parse("--served-model-name", "qwen-local").served_model_name == "qwen-local"


def test_supervised_command_removes_parent_flags_and_preserves_options() -> None:
    command = _supervised_command([
        "serve", "--model", "checkpoint with spaces", "--tensor-parallel-size", "4",
        "--tensor-parallel-rank", "0", "--tensor-parallel-supervisor",
        "--backend-option", "first=1", "--backend-option=second=true", "--supervised-child",
    ])
    assert command == [
        "serve", "--model", "checkpoint with spaces", "--tensor-parallel-size", "4",
        "--backend-option", "first=1", "--backend-option=second=true",
        "--no-tensor-parallel-supervisor", "--supervised-child",
    ]


def test_supervised_parent_does_not_construct_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSupervisor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def run(self):
            captured["ran"] = True
            return 7

    monkeypatch.setattr(cli, "TensorParallelSupervisor", FakeSupervisor)
    monkeypatch.setattr(cli, "create_backend", lambda args: pytest.fail("backend was constructed"))
    monkeypatch.setattr(cli, "select_backend", lambda args: "torch")

    assert cli.main([
        "serve", "--model", "checkpoint", "--backend", "torch", "--tensor-parallel-size", "2",
    ]) == 7
    assert captured["ran"] is True


def test_supervised_parent_selects_before_construct(monkeypatch) -> None:
    order: list[str] = []
    monkeypatch.setattr(cli, "select_backend", lambda args: order.append("select") or "torch")
    monkeypatch.setattr(cli, "create_backend", lambda args: order.append("create") or _FakeBackend())

    class FakeSupervisor:
        def __init__(self, **kwargs):
            order.append("supervisor")

        def run(self):
            return 0

    monkeypatch.setattr(cli, "TensorParallelSupervisor", FakeSupervisor)
    assert cli.main([
        "serve", "--model", "m", "--backend", "torch", "--tensor-parallel-size", "2",
    ]) == 0
    assert order == ["select", "supervisor"]


def test_supervised_parent_passes_lifecycle_options(monkeypatch) -> None:
    captured = {}

    class FakeSupervisor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return 0

    monkeypatch.setattr(cli, "TensorParallelSupervisor", FakeSupervisor)
    monkeypatch.setattr(cli, "select_backend", lambda args: "torch")
    assert cli.main([
        "serve", "--model", "m", "--backend", "torch", "--tensor-parallel-size", "3",
        "--tensor-parallel-startup-timeout", "12", "--tensor-parallel-shutdown-timeout", "4",
        "--tensor-parallel-master-addr", "127.0.0.1", "--tensor-parallel-master-port", "12345",
        "--tensor-parallel-rendezvous-dir", "/tmp/rv",
    ]) == 0
    assert captured["world_size"] == 3
    assert captured["startup_timeout"] == 12.0
    assert captured["shutdown_timeout"] == 4.0
    assert captured["master_addr"] == "127.0.0.1"
    assert captured["master_port"] == 12345
    assert captured["rendezvous_dir"] == "/tmp/rv"


def test_supervised_parent_rejects_explicit_device(monkeypatch) -> None:
    monkeypatch.setattr(cli, "select_backend", lambda args: "torch")
    with pytest.raises(Exception, match="--device"):
        cli.main([
            "serve", "--model", "checkpoint", "--backend", "torch",
            "--tensor-parallel-size", "2", "--device", "cuda:0",
        ])


def test_supervised_parent_rejects_cpp_before_backend_construction(monkeypatch) -> None:
    monkeypatch.setattr(cli, "select_backend", lambda args: "cpp")
    monkeypatch.setattr(cli, "create_backend", lambda args: pytest.fail("backend was constructed"))
    with pytest.raises(UnsupportedFeatureError, match=r"Python C\+\+ Qwen adapter"):
        cli.main([
            "serve", "--model", "checkpoint", "--backend", "cpp", "--tensor-parallel-size", "2",
        ])


def test_supervised_child_rank_zero_prepares_and_serves(monkeypatch) -> None:
    backend = _FakeBackend()
    served: list[tuple[object, dict]] = []
    monkeypatch.setattr(cli, "create_backend", lambda args: backend)
    monkeypatch.setattr(cli, "serve", lambda backend, **kwargs: served.append((backend, kwargs)))

    assert cli.main([
        "serve", "--model", "checkpoint", "--backend", "torch", "--tensor-parallel-size", "2",
        "--tensor-parallel-rank", "0", "--supervised-child",
    ]) == 0
    assert backend.prepare_calls == 1
    assert backend.worker_calls == 0
    assert len(served) == 1
    assert served[0][0] is backend
    assert served[0][1]["on_ready"] is not None


def test_supervised_nonzero_rank_uses_worker_without_http(monkeypatch) -> None:
    backend = _FakeBackend()
    served: list[bool] = []
    monkeypatch.setattr(cli, "create_backend", lambda args: backend)
    monkeypatch.setattr(cli, "serve", lambda *args, **kwargs: served.append(True))

    assert cli.main([
        "serve", "--model", "checkpoint", "--backend", "torch", "--tensor-parallel-size", "2",
        "--tensor-parallel-rank", "1", "--supervised-child",
    ]) == 0
    assert backend.worker_calls == 1
    assert backend.prepare_calls == 0
    assert served == []


def test_supervised_worker_readiness_callback_is_passed(monkeypatch) -> None:
    backend = _FakeBackend()
    callbacks: list[object] = []

    def run_worker(on_ready=None):
        callbacks.append(on_ready)
        if on_ready:
            on_ready()

    backend.run_worker = run_worker
    monkeypatch.setattr(cli, "create_backend", lambda args: backend)
    monkeypatch.setattr(cli, "_emit_readiness", lambda rank: callbacks.append(rank))
    assert cli.main([
        "serve", "--model", "m", "--backend", "torch", "--tensor-parallel-size", "2",
        "--tensor-parallel-rank", "1", "--supervised-child",
    ]) == 0
    assert callable(callbacks[0])
    assert callbacks[-1] == 1


def test_manual_rank_path_does_not_spawn(monkeypatch) -> None:
    backend = _FakeBackend()
    monkeypatch.setattr(cli, "create_backend", lambda args: backend)
    monkeypatch.setattr(cli, "serve", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "TensorParallelSupervisor", lambda **kwargs: pytest.fail("spawned"))
    assert cli.main([
        "serve", "--model", "m", "--backend", "torch", "--tensor-parallel-size", "2",
        "--no-tensor-parallel-supervisor",
    ]) == 0
    assert backend.prepare_calls == 0


def test_single_rank_path_does_not_spawn(monkeypatch) -> None:
    backend = _FakeBackend()
    monkeypatch.setattr(cli, "create_backend", lambda args: backend)
    monkeypatch.setattr(cli, "serve", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "TensorParallelSupervisor", lambda **kwargs: pytest.fail("spawned"))
    assert cli.main(["serve", "--model", "m", "--backend", "torch"]) == 0


def test_backend_options_include_supervisor_nccl_path(monkeypatch) -> None:
    monkeypatch.setenv("POCKETLLM_NCCL_ID_PATH", "/run/nccl-id")
    assert _args(_parse("--backend", "torch")).backend_options["nccl_id_path"] == "/run/nccl-id"


def test_cpp_worker_error_is_preserved_for_manual_launch(monkeypatch) -> None:
    class UnsupportedBackend(_FakeBackend):
        def run_worker(self, on_ready=None):
            raise UnsupportedFeatureError("worker unavailable")

    backend = UnsupportedBackend()
    monkeypatch.setattr(cli, "create_backend", lambda args: backend)
    with pytest.raises(UnsupportedFeatureError, match="worker unavailable"):
        cli.main([
            "serve", "--model", "m", "--backend", "torch", "--tensor-parallel-size", "2",
            "--tensor-parallel-rank", "1", "--no-tensor-parallel-supervisor",
        ])
    assert backend.health().status == "stopped"


def test_worker_failure_closes_backend(monkeypatch) -> None:
    class FailingBackend(_FakeBackend):
        def run_worker(self, on_ready=None):
            raise RuntimeError("worker boom")

    backend = FailingBackend()
    monkeypatch.setattr(cli, "create_backend", lambda args: backend)
    with pytest.raises(RuntimeError, match="worker boom"):
        cli.main([
            "serve", "--model", "m", "--backend", "torch", "--tensor-parallel-size", "2",
            "--tensor-parallel-rank", "1", "--supervised-child",
        ])
    assert backend.health().status == "stopped"


def test_backend_base_worker_hook_raises_typed_error() -> None:
    backend = BackendBase()
    with pytest.raises(Exception, match="worker entry point"):
        backend.run_worker()
