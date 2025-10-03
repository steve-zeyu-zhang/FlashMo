#!/usr/bin/env python3
"""Download official HumanML3D index, annotations, and train/val/test splits.

Original motion rotations must be downloaded separately from the AMASS portal.
This command does not download public example motions or full AMASS archives.
"""

import argparse
from pathlib import Path

from download_amass_samples import download_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/humanml_metadata"))
    args = parser.parse_args()
    download_metadata(args.output)
    print(f"Downloaded official HumanML3D metadata to {args.output}")


if __name__ == "__main__":
    main()
