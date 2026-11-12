#!/usr/bin/env python3
"""Check Python extension builds and optional CUDA compilation before GR00T loads."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import struct
import subprocess
import sys
import sysconfig
import tempfile
import time


EXTENSION = r"""
#include <Python.h>
static PyObject *info(PyObject *self, PyObject *args) {
    return Py_BuildValue("(ssi)", PY_VERSION, Py_GetVersion(), (int)sizeof(void *));
}
static PyMethodDef methods[] = {{"info", info, METH_NOARGS, NULL}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_groot_compile_probe", NULL, -1, methods};
PyMODINIT_FUNC PyInit__groot_compile_probe(void) { return PyModule_Create(&module); }
"""


def check_headers(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    package = f"libpython{sys.version_info.major}.{sys.version_info.minor}-dev"
    if manifest["package"] != package or manifest["version"].split("-")[0] != platform.python_version():
        raise RuntimeError("Python development headers do not match the running interpreter")
    runtime_package = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${Version}", f"python{sys.version_info.major}.{sys.version_info.minor}-minimal"],
        text=True, timeout=30,
    ).strip()
    if runtime_package != manifest["version"]:
        raise RuntimeError(f"Runtime package {runtime_package} != headers {manifest['version']}")
    for name, expected in manifest["header_files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Python header checksum mismatch: {name}")
    return {"root": str(root), "package": package, "version": manifest["version"],
            "deb_sha256": manifest["deb_sha256"], "verified_headers": len(manifest["header_files"])}


def check_extensions() -> list[dict]:
    results = []
    # Use the inherited CPATH, just like Triton/Inductor. Do not silently repair
    # it here: that would hide a missing export in the parent training process.
    for suffix, compiler in (("c", os.environ.get("CC", "gcc")), ("cpp", os.environ.get("CXX", "g++"))):
        with tempfile.TemporaryDirectory(prefix="groot_cpp_") as directory:
            source = Path(directory) / f"probe.{suffix}"
            binary = Path(directory) / ("_groot_compile_probe" + sysconfig.get_config_var("EXT_SUFFIX"))
            source.write_text(EXTENSION)
            command = [*shlex.split(compiler), "-shared", "-fPIC", "-O0",
                       f"-I{sysconfig.get_path('include')}", str(source), "-o", str(binary)]
            subprocess.run(command, check=True, timeout=60)
            spec = importlib.util.spec_from_file_location("_groot_compile_probe", binary)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            header_version, runtime_version, pointer_bytes = module.info()
            if header_version != platform.python_version() or pointer_bytes != struct.calcsize("P"):
                raise RuntimeError(f"Extension ABI mismatch: {module.info()}")
            results.append({"language": suffix, "compiler": compiler, "header_version": header_version,
                            "runtime_version": runtime_version, "pointer_bytes": pointer_bytes})
            print(f"PYTHON_EXTENSION_PASSED language={suffix} headers={header_version}", flush=True)
    return results


def check_cuda(devices: int) -> dict:
    import torch
    import torch._dynamo.config
    import triton

    if torch.cuda.device_count() != devices:
        raise RuntimeError(f"Expected {devices} GPUs, found {torch.cuda.device_count()}")
    torch._dynamo.config.suppress_errors = False

    def loss(value):
        return (value.sin() * value + 0.1 * value.square()).float().mean()

    compiled = torch.compile(loss, backend="inductor", fullgraph=True)
    results = []
    for index in range(devices):
        with torch.cuda.device(index):
            start = time.perf_counter()
            eager_input = torch.linspace(-1, 1, 4096, device=f"cuda:{index}", dtype=torch.bfloat16).requires_grad_()
            compiled_input = eager_input.detach().clone().requires_grad_()
            expected_loss = loss(eager_input)
            expected_gradient, = torch.autograd.grad(expected_loss, eager_input)
            actual_loss = compiled(compiled_input)
            actual_gradient, = torch.autograd.grad(actual_loss, compiled_input)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0.02, atol=1e-3)
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0.03, atol=1e-5)
            results.append({"device": index, "name": torch.cuda.get_device_name(index),
                            "seconds": time.perf_counter() - start, "loss": actual_loss.item(),
                            "finite_gradient": bool(torch.isfinite(actual_gradient).all().item())})
            print(f"CUDA_COMPILE_PASSED device={index} forward_backward=ok", flush=True)
    return {"torch": torch.__version__, "triton": triton.__version__, "devices": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers-root", type=Path, required=True)
    parser.add_argument("--cuda-devices", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {"python": sys.version, "executable": sys.executable, "cpath": os.environ.get("CPATH"),
              "status": "FAILED", "cuda_requested": args.cuda_devices}
    start = time.perf_counter()
    try:
        report["headers"] = check_headers(args.headers_root)
        report["extensions"] = check_extensions()
        if args.cuda_devices:
            report["cuda"] = check_cuda(args.cuda_devices)
        report["status"] = "COMPILE_PREFLIGHT_PASSED"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["seconds"] = time.perf_counter() - start
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
