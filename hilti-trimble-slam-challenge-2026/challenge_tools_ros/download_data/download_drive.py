#!/usr/bin/env python3
"""
Google Drive Folder Downloader

Setup:
    pip install gdown

Usage:
    python download_drive.py
    python download_drive.py -f FOLDER_ID -o ./output
"""

import argparse
from pathlib import Path

import gdown

# The folder ID from the URL
# https://drive.google.com/drive/u/3/folders/1BWFIfEL40Nvj-yeyre5O9dOiYCTWatv5
DEFAULT_FOLDER_ID = "1BWFIfEL40Nvj-yeyre5O9dOiYCTWatv5"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download files from a public Google Drive folder"
    )
    parser.add_argument(
        "--folder-id",
        "-f",
        default=DEFAULT_FOLDER_ID,
        help=f"Google Drive folder ID (default: {DEFAULT_FOLDER_ID})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./challenge_data",
        help="Output directory (default: ./challenge_data)",
    )
    args = parser.parse_args()

    folder_id = args.folder_id
    output_dir = Path(args.output)

    print("=" * 60)
    print("Google Drive Folder Downloader")
    print("=" * 60)
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the folder URL
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"

    print(f"Downloading from: {folder_url}")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    print("Starting download...")
    print("-" * 60)

    gdown.download_folder(
        url=folder_url,
        output=str(output_dir),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
        resume=True,
    )

    print("-" * 60)
    print("Download complete!")
    print(f"  Location: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
