"""
Fetches reference histopathology images from the Kaggle dataset
without downloading the entire 2.93 GB archive.

Saves chunks directly to scripts/partial_dataset.zip, resumes automatically,
and halts at ~85 MB to extract both Normal and OSCC images.

Usage:
    python scripts/fetch_reference_images.py
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from extract_partial_samples import extract_images_from_partial_zip, LOCAL_ARCHIVE


class EarlyHaltException(Exception):
    pass


def stream_partial_dataset(target_mb: float = 85.0):
    import kagglehub
    import kagglehub.clients

    target_bytes = int(target_mb * 1024 * 1024)
    orig_download_file = kagglehub.clients._download_file

    def custom_download_file(response, out_file, size_read, total_size, hash_object):
        LOCAL_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        # If out_file has existing bytes, copy or resume
        existing_size = LOCAL_ARCHIVE.stat().st_size if LOCAL_ARCHIVE.exists() else 0
        print(f"\nWriting to {LOCAL_ARCHIVE.name} (starting from {existing_size / (1024*1024):.1f} MB, target: {target_mb:.0f} MB)...")

        with open(LOCAL_ARCHIVE, "ab") as f:
            for chunk in response.iter_content(kagglehub.clients.CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                existing_size += len(chunk)
                curr_mb = existing_size / (1024 * 1024)
                print(f"\rDownloading subset: {curr_mb:.1f} MB / {target_mb:.0f} MB...", end="", flush=True)
                if existing_size >= target_bytes:
                    print(f"\n[+] Reached target threshold ({curr_mb:.1f} MB). Halting download cleanly...")
                    raise EarlyHaltException("Target sample threshold reached.")

    kagglehub.clients._download_file = custom_download_file

    try:
        kagglehub.dataset_download("ashenafifasilkebede/dataset")
    except EarlyHaltException:
        print("Download stream successfully intercepted.")
    except Exception as exc:
        print(f"Stream notice: {exc}")
    finally:
        kagglehub.clients._download_file = orig_download_file


def main():
    parser = argparse.ArgumentParser(description="Fetch and extract sample biopsy images without downloading 3 GB.")
    parser.add_argument("--target-mb", type=float, default=85.0, help="Megabytes of archive to download (default 85 MB)")
    parser.add_argument("--per-class", type=int, default=8, help="Max images to extract per class")
    args = parser.parse_args()

    # Step 1: Check if images can already be extracted from cache or unzipped files
    extracted = extract_images_from_partial_zip(LOCAL_ARCHIVE, max_per_class=args.per_class)
    if extracted >= 16:
        print("\nAll reference images already extracted! Skipping download.")
        return

    # Step 2: Stream missing chunks from Kaggle
    print(f"Connecting to Kaggle to stream up to ~{args.target_mb:.0f} MB...")
    stream_partial_dataset(target_mb=args.target_mb)

    # Step 3: Extract from the saved local archive
    print("\nExtracting reference images from local archive...")
    extract_images_from_partial_zip(LOCAL_ARCHIVE, max_per_class=args.per_class)


if __name__ == "__main__":
    main()
