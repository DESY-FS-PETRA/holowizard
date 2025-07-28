import os
import shutil
import zipfile
import requests
from pathlib import Path
from urllib.parse import urlparse
import argparse

"""
Created by claude.ai
"""


def download_file(url, local_path, chunk_size=8192):
    """Download a file from URL with progress indication"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Simple progress indicator
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading... {percent:.1f}%", end="", flush=True)

        print(f"\nDownload complete: {local_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        return False


"""
Created by claude.ai
"""


def extract_zip(zip_path, extract_to):
    """Extract ZIP file to specified directory"""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            print(f"Extracting to: {extract_to}")
            zip_ref.extractall(extract_to)

            # List extracted files
            extracted_files = zip_ref.namelist()
            print(f"Extracted {len(extracted_files)} files:")
            for file in extracted_files[:5]:  # Show first 5 files
                print(f"  - {file}")
            if len(extracted_files) > 5:
                print(f"  ... and {len(extracted_files) - 5} more files")

        return True

    except zipfile.BadZipFile:
        print("Error: Invalid ZIP file")
        return False
    except Exception as e:
        print(f"Error extracting ZIP: {e}")
        return False


"""
Created by claude.ai
"""


def get_filename_from_url(url):
    """Extract filename from URL or generate one"""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)

    # If no filename in URL, generate one
    if not filename or not filename.endswith(".zip"):
        filename = "zenodo_download.zip"

    return filename


"""
Created by claude.ai
"""


def copy_examples(dest_path):
    source_path = Path(__file__).parent.resolve() / "examples"

    # Walk through source directory
    for item in source_path.rglob("*"):
        # Calculate relative path from source
        relative_path = item.relative_to(source_path)
        dest_item = dest_path / relative_path

        try:
            if item.is_dir():
                # Create directory
                dest_item.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {relative_path}")

            elif item.is_file():
                if "__init__.py" in item.name:
                    continue
                # Create parent directory if needed
                dest_item.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(item, dest_item)  # Just copies content

                print(f"Created file: {relative_path}")

        except Exception as e:
            print(f"Error creating {relative_path}: {e}")


def download_test_data(extract_dir):
    data_url = "https://zenodo.org/api/records/15861598/files-archive"

    try:
        extract_dir.mkdir(parents=False, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")
        return 1

    filename = get_filename_from_url(data_url)
    temp_zip_path = extract_dir / filename

    print(f"Downloading test data from: {data_url}")

    # Download the file
    if not download_file(data_url, temp_zip_path):
        return 1

    print(f"Unpacking test data")

    # Extract the ZIP
    if not extract_zip(temp_zip_path, extract_dir):
        return 1

    # Clean up temporary ZIP file unless --keep-zip is specified
    try:
        temp_zip_path.unlink()
        print(f"Cleaned up temporary file: {temp_zip_path}")
    except Exception as e:
        print(f"Warning: Could not remove temporary file: {e}")

    print("\n✅ Download and extraction completed successfully!")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Creates reconstruction examples and downloads test data from Zenodo")

    parser.add_argument("directory", help="Directory to extract files to")

    args = parser.parse_args()

    print("Creating examples")

    copy_examples(args.directory)

    # Validate and create directories
    extract_dir = Path(args.directory) / "data"

    download_test_data(extract_dir)


if __name__ == "__main__":
    main()
