#!/usr/bin/env python3
"""Prepare native rotations from original AMASS files, with optional HumanML labels."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashmo.data.amass import prepare_humanml, prepare_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    samples = modes.add_parser("samples", help="Unpaired public AMASS samples for VAE/smoke checks")
    humanml = modes.add_parser("humanml", help="Official HumanML3D AMASS subset, annotations and split")
    for mode in (samples, humanml):
        mode.add_argument("--amass-root", type=Path, required=True)
        mode.add_argument("--output", type=Path, required=True)
        mode.add_argument("--fps", type=float, default=20.0)
    humanml.add_argument("--index", type=Path, required=True)
    humanml.add_argument("--texts", type=Path, required=True)
    humanml.add_argument("--split", type=Path, required=True)
    humanml.add_argument("--allow-missing", action="store_true", help="Explicitly permit a partial subset and record missing IDs")
    args = parser.parse_args()
    if args.mode == "samples":
        manifest = prepare_samples(args.amass_root, args.output, args.fps)
    else:
        manifest = prepare_humanml(args.amass_root, args.index, args.texts, args.split,
                                   args.output, args.fps, args.allow_missing)
    print(f"Prepared {manifest}")


if __name__ == "__main__":
    main()
