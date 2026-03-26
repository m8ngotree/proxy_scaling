#!/usr/bin/env python3
"""Download allenai/olmix dataset to data/olmix_release/."""
from huggingface_hub import snapshot_download


def main():
    print("Downloading allenai/olmix...")
    path = snapshot_download(
        repo_id="allenai/olmix",
        repo_type="dataset",
        local_dir="data/olmix_release",
        ignore_patterns=["*.bin", "*.pt", "*.safetensors"],
    )
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
