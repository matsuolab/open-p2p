#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.32.0",
# ]
# ///

import os
import sys
from pathlib import Path

import requests

VALID_SIZES = {"150M", "300M", "600M", "1200M"}
REPO_ID = "guaguaa/open-p2p"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"
FILENAMES = ("model_config.yaml", "checkpoint-step=00500000.ckpt")
CHUNK_SIZE = 1024 * 1024


def download_file(session: requests.Session, relative_path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{relative_path}"
    headers = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        tmp_destination = destination.with_suffix(destination.suffix + ".part")

        with tmp_destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 // total
                    print(f"\r{relative_path}: {percent:3d}%", end="", flush=True)

        tmp_destination.replace(destination)

    if total:
        print(f"\r{relative_path}: 100%")
    else:
        print(f"{relative_path}: done")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: uv run --script scripts/download_checkpoints.py <150M|300M|600M|1200M>")
        return 1

    target_size = sys.argv[1]
    if target_size not in VALID_SIZES:
        print(f"Invalid size: {target_size}. Expected one of: {', '.join(sorted(VALID_SIZES))}")
        return 1

    print(f"Downloading {target_size} from {REPO_ID}...")
    session = requests.Session()
    output_dir = Path("checkpoints") / target_size

    for filename in FILENAMES:
        relative_path = f"{target_size}/{filename}"
        download_file(session, relative_path, output_dir / filename)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
