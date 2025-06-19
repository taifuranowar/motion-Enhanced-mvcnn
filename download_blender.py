#!/usr/bin/env python3
import os
import sys
import tarfile
import subprocess
import requests
from tqdm import tqdm

def download_file(url, output_path):
    """Download a file with proper headers to avoid 403 errors"""
    print(f"Downloading Blender 3.6 from {url}...")
    
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.blender.org/download/',
        'Connection': 'keep-alive',
    }
    
    try:
        # Stream download with progress bar
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Raise error for bad responses
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(output_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))
                
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def extract_tarfile(filepath, extract_dir):
    print(f"Extracting to {extract_dir}...")
    try:
        with tarfile.open(filepath) as tar:
            tar.extractall(path=extract_dir)
        print("Extraction complete!")
        return True
    except Exception as e:
        print(f"Error extracting file: {e}")
        return False

def try_alternative_sources():
    """Try multiple sources for downloading Blender"""
    sources = [
        "https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz",
        "https://mirrors.ocf.berkeley.edu/blender/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz",
        "https://ftp.cc.uoc.gr/mirrors/blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz",
        "https://mirror.clarkson.edu/blender/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz"
    ]
    
    for url in sources:
        print(f"Trying source: {url}")
        if download_file(url, download_path):
            return True
    
    return False

def main():
    # Configuration
    global blender_dir, download_path
    blender_dir = "blender-3.6"
    blender_file = "blender-3.6.0-linux-x64.tar.xz"
    download_path = os.path.join(blender_dir, blender_file)
    
    # Create directory if it doesn't exist
    if not os.path.exists(blender_dir):
        os.makedirs(blender_dir)
        print(f"Created directory: {blender_dir}")
    
    # Download Blender if it doesn't exist
    if not os.path.exists(download_path):
        if not try_alternative_sources():
            print("\nAll download attempts failed.")
            print("Please try downloading Blender manually from: https://www.blender.org/download/")
            print(f"Save the file to: {download_path}")
            print("Then run this script again.")
            return 1
    else:
        print(f"File already exists: {download_path}")
    
    # Extract Blender
    if not extract_tarfile(download_path, blender_dir):
        return 1
    
    # Get the extracted folder path
    extracted_dir = os.path.join(blender_dir, "blender-3.6.0-linux-x64")
    blender_executable = os.path.join(extracted_dir, "blender")
    
    # Make blender executable
    os.chmod(blender_executable, 0o755)
    
    # Print instructions
    print("\n" + "="*50)
    print("BLENDER 3.6 INSTALLATION COMPLETE")
    print("="*50)
    print(f"\nBlender is installed in: {os.path.abspath(extracted_dir)}")
    print("\nTo run Blender with your MVCNN script, use:")
    print(f"{blender_executable} --background --python your_script.py")
    
    # Optionally verify the Python version
    try:
        result = subprocess.run(
            [blender_executable, "--background", "--python-expr", "import sys; print('Blender Python version:', sys.version)"],
            capture_output=True, text=True
        )
        version_info = result.stdout.strip()
        print("\nVerifying Blender's Python version:")
        print(version_info)
    except Exception as e:
        print(f"\nCould not verify Python version: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())