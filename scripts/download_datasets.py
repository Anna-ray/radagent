"""
Dataset Download and Setup Script for RadAgent v2

Downloads and prepares:
1. NIH ChestX-ray14 (112,120 images, ~45 GB)
2. Stanford CheXpert (224,316 images, ~11 GB)
3. Stanford MURA (40,561 images, ~6 GB)

Author: Rayane Aggoune
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json
import hashlib
from typing import Dict, List
import urllib.request
import zipfile
import tarfile
import shutil


def check_disk_space(required_gb: float) -> bool:
    """Check if sufficient disk space is available."""
    import shutil
    stat = shutil.disk_usage(".")
    available_gb = stat.free / (1024**3)
    print(f"Available disk space: {available_gb:.1f} GB")
    print(f"Required disk space: {required_gb:.1f} GB")
    return available_gb >= required_gb


def download_file(url: str, dest: Path, desc: str):
    """Download a file with progress bar."""
    print(f"\n[DOWNLOAD] {desc}")
    print(f"URL: {url}")
    print(f"Destination: {dest}")
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use wget if available (better for large files)
        subprocess.run(["wget", "-c", "-O", str(dest), url], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to urllib
        print("wget not found, using urllib (slower for large files)")
        urllib.request.urlretrieve(url, dest)
    
    print(f"✓ Downloaded: {dest}")


def verify_checksum(file_path: Path, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    print(f"Verifying checksum for {file_path.name}...")
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    actual_md5 = md5.hexdigest()
    
    if actual_md5 == expected_md5:
        print(f"✓ Checksum verified: {actual_md5}")
        return True
    else:
        print(f"✗ Checksum mismatch!")
        print(f"  Expected: {expected_md5}")
        print(f"  Actual:   {actual_md5}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path):
    """Extract zip or tar archive."""
    print(f"\n[EXTRACT] {archive_path.name}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    print(f"✓ Extracted to: {dest_dir}")


def download_nih_chestxray14(data_root: Path):
    """
    Download NIH ChestX-ray14 dataset.
    
    Official source: https://nihcc.app.box.com/v/ChestXray-NIHCC
    
    NOTE: Box.com requires manual download. This function provides instructions.
    """
    print("\n" + "="*80)
    print("NIH ChestX-ray14 Dataset")
    print("="*80)
    
    nih_dir = data_root / "nih"
    nih_dir.mkdir(parents=True, exist_ok=True)
    
    print("""
NIH ChestX-ray14 requires MANUAL download from Box.com:

1. Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC
2. Download these files:
   - images_001.tar.gz through images_012.tar.gz (12 files, ~45 GB total)
   - Data_Entry_2017.csv
   - train_val_list.txt
   - test_list.txt

3. Place all files in: {nih_dir}

4. Run this script again with --extract-nih flag

Expected directory structure after extraction:
{nih_dir}/
├── images/                # 112,120 PNG files (flat directory)
├── Data_Entry_2017.csv    # Master labels
├── train_val_list.txt     # Official train/val split
└── test_list.txt          # Official test split (25,596 images)
""".format(nih_dir=nih_dir))
    
    # Check if files exist
    required_files = [
        "Data_Entry_2017.csv",
        "train_val_list.txt",
        "test_list.txt"
    ]
    
    missing = [f for f in required_files if not (nih_dir / f).exists()]
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        print("Please download manually from Box.com")
        return False
    else:
        print("\n✓ All required files found!")
        
        # Check if images are extracted
        images_dir = nih_dir / "images"
        if images_dir.exists():
            num_images = len(list(images_dir.glob("*.png")))
            print(f"✓ Found {num_images:,} images in {images_dir}")
            if num_images >= 112000:
                print("✓ NIH ChestX-ray14 setup complete!")
                return True
            else:
                print(f"⚠️  Expected ~112,120 images, found {num_images}")
        else:
            print("⚠️  Images not extracted yet. Run with --extract-nih")
        
        return False


def extract_nih_archives(data_root: Path):
    """Extract NIH ChestX-ray14 tar.gz archives."""
    print("\n" + "="*80)
    print("Extracting NIH ChestX-ray14 Archives")
    print("="*80)
    
    nih_dir = data_root / "nih"
    images_dir = nih_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all tar.gz files
    archives = sorted(nih_dir.glob("images_*.tar.gz"))
    
    if not archives:
        print("✗ No images_*.tar.gz files found in", nih_dir)
        print("Please download from https://nihcc.app.box.com/v/ChestXray-NIHCC")
        return False
    
    print(f"Found {len(archives)} archives to extract")
    
    for i, archive in enumerate(archives, 1):
        print(f"\n[{i}/{len(archives)}] Extracting {archive.name}...")
        extract_archive(archive, images_dir)
    
    # Verify extraction
    num_images = len(list(images_dir.glob("*.png")))
    print(f"\n✓ Extraction complete! Total images: {num_images:,}")
    
    if num_images >= 112000:
        print("✓ NIH ChestX-ray14 ready for training!")
        return True
    else:
        print(f"⚠️  Expected ~112,120 images, found {num_images}")
        return False


def download_chexpert(data_root: Path):
    """
    Download Stanford CheXpert dataset.
    
    Official source: https://stanfordmlgroup.github.io/competitions/chexpert/
    
    NOTE: Requires registration and manual download.
    """
    print("\n" + "="*80)
    print("Stanford CheXpert Dataset")
    print("="*80)
    
    chexpert_dir = data_root / "chexpert"
    chexpert_dir.mkdir(parents=True, exist_ok=True)
    
    print("""
Stanford CheXpert requires MANUAL download after registration:

1. Visit: https://stanfordmlgroup.github.io/competitions/chexpert/
2. Register and accept data use agreement
3. Download:
   - CheXpert-v1.0-small.zip (~11 GB, 224,316 images)
   - OR CheXpert-v1.0.zip (~439 GB, full resolution - not needed for this demo)

4. Place CheXpert-v1.0-small.zip in: {chexpert_dir}

5. Run this script again with --extract-chexpert flag

Expected directory structure after extraction:
{chexpert_dir}/
├── train/
│   ├── patient00001/
│   │   ├── study1/
│   │   │   ├── view1_frontal.jpg
│   │   │   └── ...
│   └── ...
├── valid/
│   └── ...
├── train.csv
└── valid.csv
""".format(chexpert_dir=chexpert_dir))
    
    # Check if archive exists
    archive = chexpert_dir / "CheXpert-v1.0-small.zip"
    if archive.exists():
        print(f"\n✓ Found {archive.name}")
        print("Run with --extract-chexpert to extract")
        return True
    else:
        print(f"\n⚠️  {archive.name} not found")
        print("Please download from Stanford after registration")
        return False


def extract_chexpert(data_root: Path):
    """Extract CheXpert archive."""
    print("\n" + "="*80)
    print("Extracting Stanford CheXpert")
    print("="*80)
    
    chexpert_dir = data_root / "chexpert"
    archive = chexpert_dir / "CheXpert-v1.0-small.zip"
    
    if not archive.exists():
        print(f"✗ Archive not found: {archive}")
        return False
    
    extract_archive(archive, chexpert_dir)
    
    # Verify extraction
    train_csv = chexpert_dir / "CheXpert-v1.0-small" / "train.csv"
    if train_csv.exists():
        print("✓ CheXpert extraction complete!")
        return True
    else:
        print("✗ Extraction may have failed - train.csv not found")
        return False


def download_mura(data_root: Path):
    """
    Download Stanford MURA dataset.
    
    Official source: https://stanfordmlgroup.github.io/competitions/mura/
    
    NOTE: Requires registration and manual download.
    """
    print("\n" + "="*80)
    print("Stanford MURA Dataset")
    print("="*80)
    
    mura_dir = data_root / "mura"
    mura_dir.mkdir(parents=True, exist_ok=True)
    
    print("""
Stanford MURA requires MANUAL download after registration:

1. Visit: https://stanfordmlgroup.github.io/competitions/mura/
2. Register and accept data use agreement
3. Download:
   - MURA-v1.1.zip (~6 GB, 40,561 images)

4. Place MURA-v1.1.zip in: {mura_dir}

5. Run this script again with --extract-mura flag

Expected directory structure after extraction:
{mura_dir}/
├── train/
│   ├── XR_SHOULDER/
│   │   ├── patient00001/
│   │   │   ├── study1_positive/
│   │   │   │   ├── image1.png
│   │   │   │   └── ...
│   │   │   └── study2_negative/
│   │   └── ...
│   ├── XR_ELBOW/
│   └── ...
├── valid/
│   └── ...
├── train_image_paths.csv
├── train_labeled_studies.csv
├── valid_image_paths.csv
└── valid_labeled_studies.csv
""".format(mura_dir=mura_dir))
    
    # Check if archive exists
    archive = mura_dir / "MURA-v1.1.zip"
    if archive.exists():
        print(f"\n✓ Found {archive.name}")
        print("Run with --extract-mura to extract")
        return True
    else:
        print(f"\n⚠️  {archive.name} not found")
        print("Please download from Stanford after registration")
        return False


def extract_mura(data_root: Path):
    """Extract MURA archive."""
    print("\n" + "="*80)
    print("Extracting Stanford MURA")
    print("="*80)
    
    mura_dir = data_root / "mura"
    archive = mura_dir / "MURA-v1.1.zip"
    
    if not archive.exists():
        print(f"✗ Archive not found: {archive}")
        return False
    
    extract_archive(archive, mura_dir)
    
    # Verify extraction
    train_csv = mura_dir / "MURA-v1.1" / "train_image_paths.csv"
    if train_csv.exists():
        print("✓ MURA extraction complete!")
        return True
    else:
        print("✗ Extraction may have failed - train_image_paths.csv not found")
        return False


def create_dataset_manifest(data_root: Path):
    """Create a manifest file documenting dataset status."""
    manifest = {
        "nih_chestxray14": {
            "path": str(data_root / "nih"),
            "status": "not_downloaded",
            "num_images": 0,
            "size_gb": 45
        },
        "chexpert": {
            "path": str(data_root / "chexpert"),
            "status": "not_downloaded",
            "num_images": 0,
            "size_gb": 11
        },
        "mura": {
            "path": str(data_root / "mura"),
            "status": "not_downloaded",
            "num_images": 0,
            "size_gb": 6
        }
    }
    
    # Check NIH
    nih_images = data_root / "nih" / "images"
    if nih_images.exists():
        num_nih = len(list(nih_images.glob("*.png")))
        manifest["nih_chestxray14"]["num_images"] = num_nih
        manifest["nih_chestxray14"]["status"] = "ready" if num_nih >= 112000 else "partial"
    
    # Check CheXpert
    chexpert_train = data_root / "chexpert" / "CheXpert-v1.0-small" / "train"
    if chexpert_train.exists():
        num_chexpert = len(list(chexpert_train.rglob("*.jpg")))
        manifest["chexpert"]["num_images"] = num_chexpert
        manifest["chexpert"]["status"] = "ready" if num_chexpert >= 200000 else "partial"
    
    # Check MURA
    mura_train = data_root / "mura" / "MURA-v1.1" / "train"
    if mura_train.exists():
        num_mura = len(list(mura_train.rglob("*.png")))
        manifest["mura"]["num_images"] = num_mura
        manifest["mura"]["status"] = "ready" if num_mura >= 35000 else "partial"
    
    manifest_path = data_root / "dataset_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Manifest saved to: {manifest_path}")
    return manifest


def print_summary(manifest: Dict):
    """Print dataset download summary."""
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    
    for name, info in manifest.items():
        status_icon = "✓" if info["status"] == "ready" else "⚠️" if info["status"] == "partial" else "✗"
        print(f"\n{status_icon} {name.upper()}")
        print(f"   Status: {info['status']}")
        print(f"   Images: {info['num_images']:,}")
        print(f"   Path: {info['path']}")
    
    total_ready = sum(1 for info in manifest.values() if info["status"] == "ready")
    print(f"\n{total_ready}/3 datasets ready for training")
    
    if total_ready == 3:
        print("\n🎉 All datasets ready! You can now run:")
        print("   python scripts/run_federated_demo.py --nih-root data/nih --chexpert-root data/chexpert/CheXpert-v1.0-small --test-root data/chexpert/CheXpert-v1.0-small/valid")


def main():
    parser = argparse.ArgumentParser(description="Download and setup RadAgent v2 datasets")
    parser.add_argument("--data-root", type=str, default="data", help="Root directory for datasets")
    parser.add_argument("--extract-nih", action="store_true", help="Extract NIH ChestX-ray14 archives")
    parser.add_argument("--extract-chexpert", action="store_true", help="Extract CheXpert archive")
    parser.add_argument("--extract-mura", action="store_true", help="Extract MURA archive")
    parser.add_argument("--check-only", action="store_true", help="Only check dataset status, don't download")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("RadAgent v2 Dataset Download and Setup")
    print("="*80)
    print(f"Data root: {data_root.absolute()}")
    
    # Check disk space
    required_gb = 45 + 11 + 6  # NIH + CheXpert + MURA
    if not check_disk_space(required_gb):
        print("\n⚠️  WARNING: Insufficient disk space!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Extract if requested
    if args.extract_nih:
        extract_nih_archives(data_root)
    elif args.extract_chexpert:
        extract_chexpert(data_root)
    elif args.extract_mura:
        extract_mura(data_root)
    else:
        # Check/download datasets
        download_nih_chestxray14(data_root)
        download_chexpert(data_root)
        download_mura(data_root)
    
    # Create manifest and print summary
    manifest = create_dataset_manifest(data_root)
    print_summary(manifest)


if __name__ == "__main__":
    main()

# Made with Bob
