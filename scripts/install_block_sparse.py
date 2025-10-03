"""Build the pinned Block-Sparse-Attention backend in the current environment.

With --download-cuda, install NVIDIA's small compiler/runtime development
redistributions under .cache/cuda-12.4 (no system CUDA changes).
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tarfile

import requests

REVISION = "49d6c39e4dc0303442cda3bb758b3925d4399c49"
ROOT = Path(__file__).resolve().parents[1]


def download_cuda(destination):
    base = "https://developer.download.nvidia.com/compute/cuda/redist/"
    response = requests.get(base + "redistrib_12.4.1.json", timeout=60)
    response.raise_for_status()
    manifest = response.json()
    destination.mkdir(parents=True, exist_ok=True)
    for package in ("cuda_nvcc", "cuda_cudart", "cuda_cccl"):
        info = manifest[package]["linux-x86_64"]
        archive = destination.parent / Path(info["relative_path"]).name
        if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != info["sha256"]:
            with requests.get(base + info["relative_path"], stream=True, timeout=120) as result:
                result.raise_for_status()
                with archive.open("wb") as stream:
                    for chunk in result.iter_content(1024 * 1024):
                        stream.write(chunk)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != info["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {archive}")
        with tarfile.open(archive) as tar:
            for member in tar:
                parts = Path(member.name).parts
                if len(parts) < 2:
                    continue
                member.name = str(Path(*parts[1:]))
                if not (destination / member.name).resolve().is_relative_to(destination.resolve()):
                    raise RuntimeError("Unsafe CUDA archive path")
                tar.extract(member, destination, filter="data")
    # torch's extension builder accepts NVIDIA redistributions' lib/ layout.
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-cuda", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--arch", default="90", help="90=Hopper; 80=Ampere, 89=Ada")
    args = parser.parse_args()
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    if args.download_cuda:
        cuda_path = download_cuda(ROOT / ".cache/cuda-12.4")
        env["CUDA_HOME"] = str(cuda_path)
        env["PATH"] = str(cuda_path / "bin") + os.pathsep + env["PATH"]
    elif not shutil.which("nvcc") and not env.get("CUDA_HOME"):
        parser.error("nvcc is missing; pass --download-cuda or set CUDA_HOME")
    subprocess.run([sys.executable, "-m", "pip", "install", "packaging", "ninja", "wheel", "setuptools>=68", "einops"], check=True)
    # PyTorch CUDA wheels provide cuBLAS/cuSPARSE/cuSOLVER headers separately.
    # Expose these to nvcc rather than downloading another multi-GB toolkit.
    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    include_dirs = sorted(str(p) for p in nvidia.glob("*/include") if p.is_dir())
    env["CPATH"] = os.pathsep.join(include_dirs + ([env["CPATH"]] if env.get("CPATH") else []))
    checkout = ROOT / "third_party/checkouts/block_sparse_attn"
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/mit-han-lab/Block-Sparse-Attention.git", str(checkout)], check=True)
    subprocess.run(["git", "checkout", REVISION], cwd=checkout, check=True)
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=checkout, check=True)
    env.update(MAX_JOBS=str(args.jobs), BLOCK_SPARSE_ATTN_CUDA_ARCHS=args.arch,
               BLOCK_SPARSE_ATTN_FORCE_BUILD="TRUE")
    subprocess.run([sys.executable, "-m", "pip", "install", "--use-pep517", "--no-build-isolation", str(checkout)], env=env, check=True)


if __name__ == "__main__":
    main()
