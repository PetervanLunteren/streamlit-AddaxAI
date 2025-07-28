


import os
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

def download_urls(urls, download_dir="downloads", max_workers=None, ui_pbars=None, pbar_id="download", total_size="0 MB"):
    """
    Download a nested list of URLs with automatic optimization for connection speed.
    
    Args:
        urls: Nested list where each item is [url, filename]
        download_dir: Directory to save files
        max_workers: Number of concurrent downloads (auto-detected if None)
        ui_pbars: MultiProgressBars instance for UI updates
        pbar_id: ID of the progress bar in ui_pbars
        total_size: Total size as string (e.g., "104 MB") or int (bytes)
    
    Returns:
        List of successfully downloaded file paths
    """
    
    # Create download directory
    os.makedirs(download_dir, exist_ok=True)
    
    # Convert nested list to (url, filename) tuples
    download_tasks = []
    for item in urls:
        if isinstance(item, list) and len(item) >= 2:
            url, filename = item[0], item[1]
        elif isinstance(item, tuple) and len(item) >= 2:
            url, filename = item[0], item[1]
        else:
            # Fallback for other formats
            url = item if isinstance(item, str) else str(item)
            filename = os.path.basename(url.split('?')[0]) or f"file_{len(download_tasks)}"
        
        filepath = os.path.join(download_dir, filename)
        download_tasks.append((url, filepath, filename))
    
    # Test connection speed to determine optimal settings
    if max_workers is None:
        max_workers = 1  # Force single-threaded for large files from GitHub
    
    chunk_size = 1024 * 64  # Use smaller 64KB chunks for better stability
    
    print(f"Starting download of {len(download_tasks)} files with {max_workers} workers")
    
    # Parse total size
    if isinstance(total_size, str):
        total_bytes = _parse_size_string(total_size)
        print(f"Using total size: {total_size} ({total_bytes:,} bytes)")
    else:
        total_bytes = int(total_size)
        print(f"Using total size: {total_bytes:,} bytes ({total_bytes / (1024*1024):.1f} MB)")
    
    successful_downloads = []
    
    # Create overall progress bar
    pbar = tqdm(total=total_bytes, unit='B', unit_scale=True, desc="Downloading", 
                ncols=80, miniters=1, mininterval=0.1)
    pbar_lock = threading.Lock()
    
    def update_progress(bytes_downloaded):
        with pbar_lock:
            pbar.update(bytes_downloaded)
            # Update UI progress bars if provided
            if ui_pbars and pbar_id:
                ui_pbars.update_from_tqdm_object(pbar_id, pbar)
    
    try:
        if max_workers == 1:
            # Sequential downloads for slow/unstable connections
            for url, filepath, filename in download_tasks:
                if _download_single_file(url, filepath, filename, chunk_size, update_progress):
                    successful_downloads.append(filepath)
        else:
            # Concurrent downloads for fast connections
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all download tasks
                future_to_task = {
                    executor.submit(_download_single_file, url, filepath, filename, chunk_size, update_progress): (url, filepath, filename)
                    for url, filepath, filename in download_tasks
                }
                
                # Process completed downloads
                for future in as_completed(future_to_task):
                    url, filepath, filename = future_to_task[future]
                    try:
                        if future.result():
                            successful_downloads.append(filepath)
                    except Exception as e:
                        print(f"\nFailed to download {filename}: {e}")
    
    finally:
        pbar.close()
    
    print(f"\nSuccessfully downloaded {len(successful_downloads)}/{len(download_tasks)} files")
    return successful_downloads


def _parse_size_string(size_str):
    """
    Parse size string like '104 MB', '1.5 GB', '500 KB' into bytes
    
    Args:
        size_str: String representation of size (e.g., "104 MB")
    
    Returns:
        int: Size in bytes
    """
    size_str = size_str.strip().upper()
    
    # Extract number and unit using regex
    match = re.match(r'^([0-9]*\.?[0-9]+)\s*([A-Z]*B?)$', size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}. Expected format like '104 MB', '1.5 GB', etc.")
    
    number = float(match.group(1))
    unit = match.group(2)
    
    # Convert to bytes
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4,
        'K': 1024,  # Alternative
        'M': 1024 ** 2,  # Alternative
        'G': 1024 ** 3,  # Alternative
        'T': 1024 ** 4,  # Alternative
    }
    
    if unit not in multipliers:
        raise ValueError(f"Unknown unit: {unit}. Supported units: B, KB, MB, GB, TB")
    
    return int(number * multipliers[unit])


def _download_single_file(url, filepath, filename, chunk_size, progress_callback=None, max_retries=5):
    """Download a single file with resume capability and retries"""
    
    print(f"Starting download of {filename} from {url}")
    
    base_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    }
    
    partial_path = filepath + ".partial"
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries} for {filename}")
            
            headers = base_headers.copy()
            
            # Check for existing partial download
            start_byte = 0
            if os.path.exists(partial_path):
                start_byte = os.path.getsize(partial_path)
                headers['Range'] = f'bytes={start_byte}-'
                print(f"  Resuming from byte {start_byte:,}")
            
            # Make request with better timeout handling
            print(f"  Making request to {url}")
            response = requests.get(url, headers=headers, stream=True, timeout=(60, 300))  # Even longer timeouts
            print(f"  Response status: {response.status_code}")
            response.raise_for_status()
            
            # Check if we got the expected response for range requests
            if start_byte > 0:
                if response.status_code == 206:
                    print(f"  Server supports resume (206)")
                elif response.status_code == 200:
                    print(f"  Server doesn't support resume, restarting download (200)")
                    start_byte = 0
                    # Remove partial file and start fresh
                    if os.path.exists(partial_path):
                        os.remove(partial_path)
            
            # Download file
            mode = 'ab' if start_byte > 0 and response.status_code == 206 else 'wb'
            print(f"  Writing to {partial_path} in mode {mode}")
            
            bytes_downloaded = 0
            last_progress_time = time.time()
            stall_timeout = 60  # 60 seconds without progress = stalled
            progress_check_interval = 5  # Check every 5 seconds
            last_check_time = time.time()
            
            with open(partial_path, mode) as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    current_time = time.time()
                    
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        last_progress_time = current_time
                        if progress_callback:
                            progress_callback(len(chunk))
                    
                    # Periodic stall check (don't check every chunk for performance)
                    if current_time - last_check_time > progress_check_interval:
                        if current_time - last_progress_time > stall_timeout:
                            raise Exception(f"Download stalled - no data for {stall_timeout} seconds")
                        last_check_time = current_time
            
            print(f"  Downloaded {bytes_downloaded:,} bytes for {filename}")
            
            # Check if we actually downloaded something (unless resuming)
            if bytes_downloaded == 0 and start_byte == 0:
                raise Exception("No data downloaded")
            
            # Verify the file exists and has reasonable size
            if not os.path.exists(partial_path):
                raise Exception("Partial file disappeared")
                
            current_size = os.path.getsize(partial_path)
            print(f"  File size on disk: {current_size:,} bytes")
            
            # Move completed file to final location
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(partial_path, filepath)
            
            final_size = os.path.getsize(filepath)
            print(f"Successfully downloaded {filename} ({final_size:,} bytes)")
            return True
            
        except Exception as e:
            print(f"  Download attempt {attempt + 1} failed for {filename}: {e}")
            print(f"  Exception type: {type(e).__name__}")
            
            if attempt < max_retries - 1:
                retry_delay = min(2 ** attempt, 30)  # Cap at 30 seconds
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                
                # Update start_byte for next attempt if partial file exists
                if os.path.exists(partial_path):
                    new_start_byte = os.path.getsize(partial_path)
                    print(f"  Partial file size: {new_start_byte:,} bytes")
            else:
                print(f"Failed to download {filename} after {max_retries} attempts")
                # Cleanup partial file
                if os.path.exists(partial_path):
                    os.remove(partial_path)
                    print(f"  Cleaned up partial file")
                return False
    
    return False

