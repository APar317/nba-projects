"""
Extracts intact images from a partial or fully downloaded Kaggle/Mendeley zip archive
without needing to download the full 3 GB dataset.

Handles standard ZIP files as well as streamed ZIP files where compressed_size is 0
in the local file header (data descriptor format).

Usage:
    python scripts/extract_partial_samples.py [optional_path_to_archive]
"""

import json
import os
import shutil
import struct
import sys
import zlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "static" / "samples"
NORMAL_DIR = SAMPLES_DIR / "normal"
OSCC_DIR = SAMPLES_DIR / "oscc"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"

LOCAL_ARCHIVE = BASE_DIR / "scripts" / "partial_dataset.zip"
CACHE_DIR = Path.home() / ".cache" / "kagglehub" / "datasets" / "ashenafifasilkebede" / "dataset"
DEFAULT_ARCHIVE = CACHE_DIR / "1.archive"


def find_archive():
    # 1. Local copy in scripts/
    if LOCAL_ARCHIVE.exists() and LOCAL_ARCHIVE.stat().st_size > 1024 * 1024:
        return LOCAL_ARCHIVE

    # 2. Check default kagglehub archive
    if DEFAULT_ARCHIVE.exists():
        return DEFAULT_ARCHIVE

    # 3. Check for any other archive or zip in kagglehub cache
    if CACHE_DIR.exists():
        for item in CACHE_DIR.rglob("*"):
            if item.is_file() and (item.suffix in (".archive", ".zip") or "archive" in item.name):
                if item.stat().st_size > 1024 * 1024:
                    return item
    return DEFAULT_ARCHIVE


def check_and_copy_extracted_files(max_per_class: int = 8):
    """If kagglehub already unzipped files into its cache directory, copy them directly."""
    if not CACHE_DIR.exists():
        return 0

    normal_files = sorted(list(CACHE_DIR.rglob("*Normal*/*.jpg")) + list(CACHE_DIR.rglob("*normal*/*.jpg")))
    oscc_files = sorted(list(CACHE_DIR.rglob("*OSCC*/*.jpg")) + list(CACHE_DIR.rglob("*oscc*/*.jpg")))

    if not (normal_files or oscc_files):
        return 0

    print(f"Found loose files in cache: {len(normal_files)} Normal, {len(oscc_files)} OSCC")
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    OSCC_DIR.mkdir(parents=True, exist_ok=True)

    manifest_images = []
    for f in normal_files[:max_per_class]:
        dest = NORMAL_DIR / f.name
        shutil.copy2(f, dest)
        manifest_images.append({"file": f"normal/{f.name}", "label": "Normal"})
        print(f"  [+] Copied Normal: {f.name}")

    for f in oscc_files[:max_per_class]:
        dest = OSCC_DIR / f.name
        shutil.copy2(f, dest)
        manifest_images.append({"file": f"oscc/{f.name}", "label": "OSCC"})
        print(f"  [+] Copied OSCC: {f.name}")

    if manifest_images:
        update_manifest(manifest_images)
        return len(manifest_images)
    return 0


def update_manifest(new_images):
    current_manifest = {}
    if MANIFEST_PATH.exists():
        try:
            current_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    dataset_meta = current_manifest.get("dataset", {
        "name": "Histopathological Imaging Database for Oral Cancer Analysis",
        "authors": "Rahman T.Y., Mahanta L.B., Das A.K., Sarma J.D.",
        "version": "2",
        "doi": "10.17632/ftmp4cvtmb.2",
        "source_url": "https://data.mendeley.com/datasets/ftmp4cvtmb/2",
        "license": "CC BY 4.0",
    })

    existing_files = {entry["file"] for entry in current_manifest.get("images", [])}
    merged_images = list(current_manifest.get("images", []))
    for img_entry in new_images:
        if img_entry["file"] not in existing_files:
            merged_images.append(img_entry)

    MANIFEST_PATH.write_text(json.dumps({"dataset": dataset_meta, "images": merged_images}, indent=2), encoding="utf-8")
    print(f"\nManifest updated at {MANIFEST_PATH} with {len(merged_images)} total reference images.")


def extract_images_from_partial_zip(archive_path: Path = None, max_per_class: int = 8):
    if archive_path is None or not archive_path.exists():
        archive_path = find_archive()

    # Check if loose images already exist in cache
    copied = check_and_copy_extracted_files(max_per_class=max_per_class)
    if copied > 0:
        return copied

    if not archive_path.exists():
        print(f"Archive not found at: {archive_path}")
        return 0

    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"Reading archive: {archive_path} ({size_mb:.1f} MB)...")

    with open(archive_path, "rb") as f:
        data = f.read()

    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    OSCC_DIR.mkdir(parents=True, exist_ok=True)

    total_len = len(data)
    extracted_normal = 0
    extracted_oscc = 0
    manifest_images = []

    header_indices = []
    scan_pos = 0
    while True:
        pos = data.find(b"PK\x03\x04", scan_pos)
        if pos == -1:
            break
        header_indices.append(pos)
        scan_pos = pos + 4

    print(f"Found {len(header_indices)} file headers in archive.")

    for i, idx in enumerate(header_indices):
        if idx + 30 > total_len:
            break

        comp_method = struct.unpack("<H", data[idx + 8 : idx + 10])[0]
        c_size = struct.unpack("<I", data[idx + 18 : idx + 22])[0]
        fn_len = struct.unpack("<H", data[idx + 26 : idx + 28])[0]
        extra_len = struct.unpack("<H", data[idx + 28 : idx + 30])[0]

        fn_start = idx + 30
        fn_end = fn_start + fn_len
        file_data_start = fn_end + extra_len

        filename = data[fn_start:fn_end].decode("utf-8", errors="ignore").replace("\\", "/")
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        if c_size > 0:
            file_data_end = file_data_start + c_size
        else:
            next_header = header_indices[i + 1] if i + 1 < len(header_indices) else total_len
            dd_pos = data.rfind(b"PK\x07\x08", file_data_start, next_header)
            file_data_end = dd_pos if dd_pos != -1 else next_header

        if file_data_start >= total_len:
            break

        raw_chunk = data[file_data_start:file_data_end]
        file_bytes = None

        if raw_chunk.startswith(b"\xff\xd8\xff") or raw_chunk.startswith(b"\x89PNG"):
            file_bytes = raw_chunk
        else:
            for wbits in (-zlib.MAX_WBITS, zlib.MAX_WBITS, zlib.MAX_WBITS | 16, zlib.MAX_WBITS | 32):
                try:
                    d_obj = zlib.decompressobj(wbits)
                    decomp = d_obj.decompress(raw_chunk)
                    if decomp.startswith(b"\xff\xd8\xff") or decomp.startswith(b"\x89PNG") or len(decomp) > 1000:
                        file_bytes = decomp
                        break
                except Exception:
                    continue

        if not file_bytes:
            continue

        basename = Path(filename).name
        is_normal = "normal" in filename.lower()
        is_oscc = "oscc" in filename.lower()

        if is_normal and extracted_normal < max_per_class:
            dest = NORMAL_DIR / basename
            dest.write_bytes(file_bytes)
            extracted_normal += 1
            manifest_images.append({"file": f"normal/{basename}", "label": "Normal"})
            print(f"  [+] Extracted Normal: {basename} ({len(file_bytes) // 1024} KB)")

        elif is_oscc and extracted_oscc < max_per_class:
            dest = OSCC_DIR / basename
            dest.write_bytes(file_bytes)
            extracted_oscc += 1
            manifest_images.append({"file": f"oscc/{basename}", "label": "OSCC"})
            print(f"  [+] Extracted OSCC: {basename} ({len(file_bytes) // 1024} KB)")

    if manifest_images:
        update_manifest(manifest_images)

    print(f"\nCompleted! Extracted {extracted_normal} Normal and {extracted_oscc} OSCC sample images.")
    return extracted_normal + extracted_oscc


if __name__ == "__main__":
    archive = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    extract_images_from_partial_zip(archive)
