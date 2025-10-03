#!/usr/bin/env python3
"""Download two author-published AMASS examples and optional HumanML metadata.

AMASS research terms: https://amass.is.tue.mpg.de/license.html . Samples are
from the official AMASS repository, not redistributed full dataset archives.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile


AMASS_COMMIT = "36d86b55c7b90870dcdebe10b14ece99ddcc7dfc"
HUMANML_COMMIT = "9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23"
SAMPLES = {
    "amass_sample.npz": "b7bd737171d3ac9c7b5e6423db0aec40fe532a4dd0004deeb566f246ee8eddf6",
    "dmpl_sample.npz": "fe2456291863c950eeffd5f0c25a2102f2861ee14301367b16c13654dfaa9afa",
}


def download(url, destination, expected_sha256=None, max_bytes=64 * 1024 * 1024):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        content = destination.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise ValueError(f"Download exceeds size limit: {url}")
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {destination}; expected {expected_sha256}, got {digest}")
    destination.write_bytes(content)
    return {"url": url, "path": str(destination), "sha256": digest, "bytes": len(content)}


def download_metadata(output):
    records = []
    base = f"https://raw.githubusercontent.com/EricGuo5513/HumanML3D/{HUMANML_COMMIT}"
    for relative in ["index.csv", "HumanML3D/train.txt", "HumanML3D/val.txt", "HumanML3D/test.txt",
                     "HumanML3D/train_val.txt", "HumanML3D/all.txt", "HumanML3D/texts.zip"]:
        destination = output / Path(relative).name
        records.append(download(f"{base}/{relative}", destination))
    # Extract only text annotations, rejecting traversal and unexpected content.
    with zipfile.ZipFile(io.BytesIO((output / "texts.zip").read_bytes())) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe path in HumanML3D metadata archive")
            if member.is_dir() or name.suffix != ".txt" or "__MACOSX" in name.parts:
                continue
            if member.file_size > 1024 * 1024:
                raise ValueError("Unexpectedly large caption file")
            destination = output / "texts" / name.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
    (output / "provenance.json").write_text(json.dumps({"repository": "EricGuo5513/HumanML3D",
                "commit": HUMANML_COMMIT, "files": records}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/amass_samples"))
    parser.add_argument("--humanml-metadata", type=Path, help="Also download official index, splits, and text annotations")
    parser.add_argument("--metadata-only", action="store_true", help="Download only HumanML3D metadata, without public AMASS examples")
    args = parser.parse_args()
    if args.metadata_only and args.humanml_metadata is None:
        parser.error("--metadata-only requires --humanml-metadata DIRECTORY")
    if not args.metadata_only:
        records = []
        for name, digest in SAMPLES.items():
            url = f"https://raw.githubusercontent.com/nghorbani/amass/{AMASS_COMMIT}/support_data/github_data/{name}"
            records.append(download(url, args.output / name, digest))
        provenance = {"repository": "nghorbani/amass", "commit": AMASS_COMMIT,
                      "license": "https://amass.is.tue.mpg.de/license.html",
                      "humanml_subset_verified": False, "paired_text_available": False, "files": records}
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"Downloaded {len(records)} authentic public AMASS examples to {args.output}; no paired HumanML captions")
    if args.humanml_metadata:
        download_metadata(args.humanml_metadata)
        print(f"Downloaded official HumanML3D metadata to {args.humanml_metadata}")


if __name__ == "__main__":
    main()
