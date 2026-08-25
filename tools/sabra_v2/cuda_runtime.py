"""P27R1 host-driver-first CUDA child environment and fail-closed probe."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping


HOST_DRIVER_DIRECTORY = Path("/usr/lib/x86_64-linux-gnu")
RECOVERY_VERSION = "HOST_DRIVER_FIRST_V1"
EXPECTED_TORCH = "2.5.1+cu121"
EXPECTED_TORCH_CUDA = "12.1"
FORBIDDEN_LIBRARY_COMPONENTS = frozenset({"compat", "stubs"})


def _has_forbidden_component(value: str) -> bool:
    return any(component.lower() in FORBIDDEN_LIBRARY_COMPONENTS for component in Path(value).parts)


def build_p27_cuda_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an explicit child environment with the real host driver first."""
    environment = dict(os.environ if source is None else source)
    retained: list[str] = []
    seen: set[str] = set()
    host = str(HOST_DRIVER_DIRECTORY)
    for component in environment.get("LD_LIBRARY_PATH", "").split(":"):
        component = component.strip()
        if not component or component == host or _has_forbidden_component(component) or component in seen:
            continue
        seen.add(component)
        retained.append(component)
    environment["LD_LIBRARY_PATH"] = ":".join([host, *retained])
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["P27_CUDA_RUNTIME_RECOVERY"] = RECOVERY_VERSION
    return environment


def _loaded_libcuda_paths() -> list[str]:
    ctypes.CDLL("libcuda.so.1")
    paths: set[str] = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        candidate = line.rsplit(None, 1)[-1]
        if "/" not in candidate or not Path(candidate).name.startswith("libcuda.so"):
            continue
        paths.add(candidate)
    return sorted(paths)


def validate_resolved_libcuda(paths: list[str]) -> Path:
    if not paths:
        raise RuntimeError("libcuda.so.1 was not mapped")
    for value in paths:
        if _has_forbidden_component(value):
            component = next(part for part in Path(value).parts if part.lower() in FORBIDDEN_LIBRARY_COMPONENTS)
            raise RuntimeError(f"resolved libcuda has forbidden {component} precedence: {value}")
    resolved = {Path(value).resolve(strict=True) for value in paths}
    if len(resolved) != 1:
        raise RuntimeError(f"multiple libcuda driver objects were mapped: {sorted(map(str, resolved))}")
    selected = next(iter(resolved))
    try:
        selected.relative_to(HOST_DRIVER_DIRECTORY.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"libcuda did not resolve to verified host driver directory: {selected}") from exc
    return selected


def probe_current_cuda() -> dict[str, object]:
    libcuda = validate_resolved_libcuda(_loaded_libcuda_paths())
    import torch

    available = torch.cuda.is_available()
    result: dict[str, object] = {
        "schema_version": "P27R1_CUDA_RUNTIME_PROBE_V1",
        "status": "PASS" if available else "FAIL",
        "runtime_recovery": os.environ.get("P27_CUDA_RUNTIME_RECOVERY"),
        "path": os.environ.get("PATH"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "cuda_path": os.environ.get("CUDA_PATH"),
        "resolved_libcuda": str(libcuda),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_is_available": available,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "gpu_vram": torch.cuda.get_device_properties(0).total_memory if available else None,
    }
    if torch.__version__ != EXPECTED_TORCH or torch.version.cuda != EXPECTED_TORCH_CUDA:
        raise RuntimeError(f"frozen PyTorch runtime changed: {torch.__version__}, CUDA {torch.version.cuda}")
    if not available:
        raise RuntimeError("CUDA is unavailable after host-driver-first recovery")
    return result


def probe_cuda_subprocess(environment: Mapping[str, str], python: str | Path = sys.executable) -> dict[str, object]:
    completed = subprocess.run(
        [str(python), "-m", "tools.sabra_v2.cuda_runtime", "--probe"],
        env=dict(environment),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"P27R1 CUDA child probe failed ({completed.returncode}): stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )
    payload = json.loads(completed.stdout)
    if payload.get("status") != "PASS":
        raise RuntimeError(f"P27R1 CUDA child probe did not pass: {payload}")
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", required=True)
    return parser


def main() -> None:
    make_parser().parse_args()
    print(json.dumps(probe_current_cuda(), sort_keys=True))


if __name__ == "__main__":
    main()
