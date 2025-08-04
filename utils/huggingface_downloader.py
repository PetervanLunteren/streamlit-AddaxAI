import os
import time
import requests
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor  # as_completed - UNUSED: Vulture detected unused import
# from urllib.parse import urlparse  # UNUSED: Vulture detected unused import
from tqdm import tqdm
from huggingface_hub import HfApi  # hf_hub_download - UNUSED: Vulture detected unused import
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError


class HuggingFaceRepoDownloader:
    def __init__(self, max_workers=4, chunk_size=8192, timeout=30):
        """
        Initialize the Hugging Face repository downloader.
        
        Args:
            max_workers (int): Maximum number of concurrent downloads
            chunk_size (int): Size of chunks for file downloads (bytes)
            timeout (int): Request timeout in seconds
        """
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.api = HfApi()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HuggingFace-Repo-Downloader/1.0'
        })
        
        # Adaptive scaling parameters
        self.min_workers = 1
        self.max_workers_limit = 16
        self.speed_samples = []
        self.max_speed_samples = 10
        self.last_adjustment_time = 0
        self.adjustment_interval = 10  # seconds
        
        # Progress tracking
        # self.total_bytes = 0  # UNUSED: Vulture detected unused attribute
        # self.downloaded_bytes = 0  # UNUSED: Vulture detected unused attribute
        self.pbar = None
        self.lock = threading.Lock()
    
    def get_repo_info(self, repo_id, revision="main"):
        """
        Get repository information including total size and file list.
        
        Args:
            repo_id (str): Repository ID (e.g., "username/repo-name")
            revision (str): Branch or revision to download
            
        Returns:
            tuple: (total_size_bytes, files_info_list)
        """
        try:
            print(f"📋 Fetching repository info for {repo_id}...")
            
            # Get repository files
            files = self.api.list_repo_files(
                repo_id=repo_id,
                revision=revision,
                repo_type="model"  # Can be "model", "dataset", or "space"
            )
            
            # Get detailed file information
            files_info = []
            total_size = 0
            
            print(f"📊 Calculating total size for {len(files)} files...")
            
            for file_path in tqdm(files, desc="Analyzing files", leave=False):
                try:
                    # Get file info using the API
                    file_info = self.api.get_paths_info(
                        repo_id=repo_id,
                        paths=[file_path],
                        revision=revision,
                        repo_type="model"
                    )[0]
                    
                    if hasattr(file_info, 'size') and file_info.size:
                        file_size = file_info.size
                        total_size += file_size
                        files_info.append({
                            'path': file_path,
                            'size': file_size,
                            'url': f"https://huggingface.co/{repo_id}/resolve/{revision}/{file_path}"
                        })
                
                except Exception as e:
                    print(f"⚠️  Could not get size for {file_path}: {e}")
                    # Add file without size info
                    files_info.append({
                        'path': file_path,
                        'size': 0,
                        'url': f"https://huggingface.co/{repo_id}/resolve/{revision}/{file_path}"
                    })
            
            return total_size, files_info
            
        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            raise ValueError(f"Repository not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Error fetching repository info: {e}")
    
    def update_progress(self, bytes_downloaded):
        """Update the progress bar thread-safely (no UI updates from threads)."""
        with self.lock:
            # self.downloaded_bytes += bytes_downloaded  # UNUSED: Vulture detected unused attribute
            if self.pbar:
                self.pbar.update(bytes_downloaded)
    
    def measure_download_speed(self, start_time, bytes_downloaded):
        """Measure and record download speed for adaptive scaling."""
        if bytes_downloaded > 0:
            elapsed = time.time() - start_time
            if elapsed > 0: 
                speed = bytes_downloaded / elapsed  # bytes per second
                with self.lock:
                    self.speed_samples.append(speed)
                    if len(self.speed_samples) > self.max_speed_samples:
                        self.speed_samples.pop(0)
    
    def adjust_workers(self):
        """Dynamically adjust the number of workers based on performance."""
        current_time = time.time()
        if current_time - self.last_adjustment_time < self.adjustment_interval:
            return
        
        with self.lock:
            if len(self.speed_samples) < 3:
                return
            
            avg_speed = sum(self.speed_samples) / len(self.speed_samples)
            recent_speed = sum(self.speed_samples[-3:]) / 3
            
            # If recent speed is significantly lower, reduce workers
            if recent_speed < avg_speed * 0.7 and self.max_workers > self.min_workers:
                self.max_workers = max(self.min_workers, self.max_workers - 1)
                print(f"🔽 Reduced workers to {self.max_workers} (slow connection detected)")
            
            # If recent speed is good and stable, consider increasing workers
            elif recent_speed > avg_speed * 1.2 and self.max_workers < self.max_workers_limit:
                self.max_workers = min(self.max_workers_limit, self.max_workers + 1)
                print(f"🔼 Increased workers to {self.max_workers} (fast connection detected)")
            
            self.last_adjustment_time = current_time
    
    def download_file(self, file_info, local_dir):
        """
        Download a single file with progress tracking.
        
        Args:
            file_info (dict): File information including path, size, and URL
            local_dir (str): Local directory to save the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        file_path = file_info['path']
        file_size = file_info['size']
        file_url = file_info['url']
        
        local_file_path = os.path.join(local_dir, file_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Skip if file already exists and has correct size
        if os.path.exists(local_file_path):
            existing_size = os.path.getsize(local_file_path)
            if existing_size == file_size:
                self.update_progress(file_size)
                return True
        
        start_time = time.time()
        downloaded = 0
        
        try:
            with self.session.get(file_url, stream=True, timeout=self.timeout) as response:
                response.raise_for_status()
                
                with open(local_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            chunk_size = len(chunk)
                            downloaded += chunk_size
                            self.update_progress(chunk_size)
            
            self.measure_download_speed(start_time, downloaded)
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {file_path}: {e}")
            # Clean up partial file
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
            return False
    
    def download_repo(self, model_ID, local_dir, ui_pbars, pbar_id):
        """
        Download entire Hugging Face repository.
        
        Args:
            repo_url_or_id (str): Repository URL or ID
            local_dir (str): Local directory to save files
            revision (str): Branch or revision to download
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Parse repository ID
            repo_id = "Addax-Data-Science/" + model_ID.strip()
            revision="main"
            
            print(f"🚀 Starting download of {repo_id} (revision: {revision})")
            
            # Get repository info and total size
            total_size, files_info = self.get_repo_info(repo_id, revision)
            # self.total_bytes = total_size  # UNUSED: Vulture detected unused attribute
            
            print(f"📦 Repository size: {total_size / (1024*1024*1024):.2f} GB")
            print(f"📁 Files to download: {len(files_info)}")
            
            # Create local directory (use the specified directory directly)
            repo_local_dir = local_dir
            os.makedirs(repo_local_dir, exist_ok=True)
            
            # Initialize progress bar (no unit_scale, keep raw bytes)
            self.pbar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=False,
            )
            
            # Initialize UI progress bar
            if ui_pbars and pbar_id:
                ui_pbars.set_max_value(pbar_id, total_size)
                ui_pbars.start_pbar(pbar_id)
            
            # Download files with thread pool
            successful_downloads = 0
            failed_downloads = 0
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all download tasks
                future_to_file = {
                    executor.submit(self.download_file, file_info, repo_local_dir): file_info
                    for file_info in files_info
                }
                
                # Process completed downloads with frequent UI updates
                while future_to_file:
                    # Check for completed downloads (with timeout)
                    completed_futures = []
                    for future in list(future_to_file.keys()):
                        try:
                            # Check if future is done without blocking
                            if future.done():
                                completed_futures.append(future)
                        except:
                            pass
                    
                    # Process any completed futures
                    for future in completed_futures:
                        file_info = future_to_file.pop(future)
                        try:
                            success = future.result()
                            if success:
                                successful_downloads += 1
                            else:
                                failed_downloads += 1
                        except Exception as e:
                            print(f"❌ Unexpected error downloading {file_info['path']}: {e}")
                            failed_downloads += 1
                    
                    # Update UI progress bar from main thread
                    if ui_pbars and pbar_id and self.pbar:
                        ui_pbars.update_from_tqdm_object(pbar_id, self.pbar)
                    
                    # Periodically adjust workers based on performance
                    self.adjust_workers()
                    
                    # Short sleep to prevent busy waiting
                    time.sleep(0.1)
            
            # Close progress bar
            self.pbar.close()
            
            # Final UI update to show completion
            if ui_pbars and pbar_id and self.pbar:
                ui_pbars.update_from_tqdm_object(pbar_id, self.pbar)
            
            # Summary
            print(f"\n✅ Download completed!")
            print(f"📊 Success: {successful_downloads} files")
            if failed_downloads > 0:
                print(f"❌ Failed: {failed_downloads} files")
            print(f"📁 Files saved to: {repo_local_dir}")
            
            return failed_downloads == 0
            
        except Exception as e:
            if self.pbar:
                self.pbar.close()
            # Final UI update even on failure
            if ui_pbars and pbar_id and self.pbar:
                ui_pbars.update_from_tqdm_object(pbar_id, self.pbar)
            print(f"💥 Download failed: {e}")
            return False
