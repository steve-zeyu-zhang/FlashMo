"""Fetch exact upstream source snapshots for audit; installs nothing.

Repositories are shallow and detached at third_party/upstream.json revisions.
Existing checkouts are never reset or overwritten. Core FlashMo runtime uses
the local implementation and does not depend on these optional checkouts.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "third_party/upstream.json"


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE).stdout.strip()


def verify_license(entry, checkout):
    source = checkout / entry["license_source"]
    bundled = ROOT / "third_party" / entry["license_file"]
    expected = entry["license_sha256"]
    for path in (source, bundled):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"License SHA256 mismatch: {path}")


def inspect_checkout(name, entry, destination, submodules=False):
    commit = git(["rev-parse", "HEAD"], destination)
    if commit != entry["commit"]:
        raise RuntimeError(f"{name}: existing HEAD {commit} differs from pin {entry['commit']}; refusing to reset")
    status = git(["status", "--porcelain", "--untracked-files=no"], destination)
    if status:
        raise RuntimeError(f"{name}: tracked checkout files have local changes; refusing to modify")
    verify_license(entry, destination)
    if submodules:
        for path, info in entry.get("submodules", {}).items():
            subdirectory = destination / path
            found = git(["rev-parse", "HEAD"], subdirectory)
            if found != info["commit"]:
                raise RuntimeError(f"{name}/{path}: expected submodule {info['commit']}, got {found}")
            verify_license(info, subdirectory)
    return {"name": name, "commit": commit, "path": str(destination), "license_verified": True}


def main(argv=None):
    manifest = json.loads(MANIFEST.read_text())
    repositories = manifest["repositories"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=sorted(repositories),
                        help="Default: all implementation repositories; ZipVL website is not fetched")
    parser.add_argument("--output-root", type=Path, default=ROOT / "third_party/checkouts")
    parser.add_argument("--submodules", action="store_true", help="Fetch pinned build dependencies such as CUTLASS")
    parser.add_argument("--check", action="store_true", help="Verify existing snapshots and licenses without network access")
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    if not args.check:
        output_root.mkdir(parents=True, exist_ok=True)
    for name in args.only or repositories:
        entry = repositories[name]
        if not re.fullmatch(r"[0-9a-f]{40}", entry["commit"]):
            raise ValueError(f"{name}: revision must be a full SHA1")
        destination = output_root / name
        if destination.exists():
            if not (destination / ".git").exists():
                raise RuntimeError(f"{destination} exists but is not an expected Git checkout")
            # Verify before any operation; existing checkouts are read-only.
            report = inspect_checkout(name, entry, destination, submodules=args.submodules)
        elif args.check:
            raise FileNotFoundError(f"Missing pinned checkout: {destination}")
        else:
            destination.mkdir()
            git(["init", "--quiet"], destination)
            git(["remote", "add", "origin", entry["url"]], destination)
            git(["fetch", "--depth=1", "--no-tags", "origin", entry["commit"]], destination)
            git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], destination)
            if args.submodules and entry.get("submodules"):
                git(["submodule", "update", "--init", "--recursive", "--depth=1"], destination)
            report = inspect_checkout(name, entry, destination, submodules=args.submodules)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Upstream snapshot error: {error}", file=sys.stderr)
        raise SystemExit(1)
