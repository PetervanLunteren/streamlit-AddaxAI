"""
Video Utilities for AddaxAI

Utilities for handling video files including metadata extraction and GPS data.

Created by Peter van Lunteren for AddaxAI video support
"""

import subprocess
import json
from hachoir.metadata import extractMetadata
from hachoir.parser import createParser


def get_video_gps(file_path):
    """
    Extract GPS coordinates from video file metadata.
    
    Tries multiple methods:
    1. exiftool (most reliable for ffmpeg-added GPS data)
    2. hachoir as fallback
    
    Args:
        file_path: Path to video file
        
    Returns:
        tuple: (latitude, longitude) or None if no GPS data found
    """
    
    def parse_dms_to_decimal(dms_str):
        """Parse degrees/minutes/seconds format to decimal degrees"""
        try:
            # Format: "51 deg 21' 20.30" N" 
            parts = dms_str.strip().split()
            degrees = float(parts[0])
            minutes = float(parts[2].replace("'", ""))
            seconds = float(parts[3].replace('"', ''))
            direction = parts[4] if len(parts) > 4 else parts[-1]
            
            decimal = degrees + minutes/60 + seconds/3600
            
            # Apply direction
            if direction.upper() in ['S', 'W']:
                decimal = -decimal
                
            return decimal
        except:
            return None
    
    # Method 1: Try exiftool (best for ffmpeg GPS data)
    try:
        cmd = ['exiftool', '-json', '-gps*', '-q', str(file_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data and len(data) > 0:
                metadata = data[0]
                
                # Look for GPS coordinates in various formats
                lat = None
                lon = None
                
                # Method 1a: GPSLatitude/GPSLongitude in DMS format
                if 'GPSLatitude' in metadata and 'GPSLongitude' in metadata:
                    lat = parse_dms_to_decimal(metadata['GPSLatitude'])
                    lon = parse_dms_to_decimal(metadata['GPSLongitude'])
                
                # Method 1b: Direct decimal coordinates
                if lat is None or lon is None:
                    for lat_key in ['GPSLatitude', 'latitude', 'Latitude']:
                        for lon_key in ['GPSLongitude', 'longitude', 'Longitude']:
                            if lat_key in metadata and lon_key in metadata:
                                try:
                                    lat_val = metadata[lat_key]
                                    lon_val = metadata[lon_key]
                                    if isinstance(lat_val, (int, float)) and isinstance(lon_val, (int, float)):
                                        lat = float(lat_val)
                                        lon = float(lon_val)
                                        break
                                except:
                                    pass
                        if lat is not None:
                            break
                
                if lat is not None and lon is not None:
                    return (lat, lon)
                    
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    except Exception:
        pass
    
    # Method 2: Fallback to hachoir (for other metadata formats)
    def safe_has_key(metadata, key):
        try:
            return metadata.has(key)
        except:
            return False

    def safe_get_value(metadata, key, index=0):
        try:
            if safe_has_key(metadata, key):
                return metadata.get(key, index)
            return None
        except:
            return None
    
    try:
        parser = createParser(str(file_path))
        if not parser:
            return None

        with parser:
            metadata = extractMetadata(parser)
            if not metadata:
                return None

            # Try various GPS-related keys
            gps_keys_to_try = [
                ('location', None),
                ('latitude', 'longitude'),
                ('gps_latitude', 'gps_longitude'),
                ('coordinates', None),
                ('position', None),
            ]
            
            for key_info in gps_keys_to_try:
                if len(key_info) == 2 and key_info[1] is not None:
                    lat_key, lon_key = key_info
                    lat = safe_get_value(metadata, lat_key)
                    lon = safe_get_value(metadata, lon_key)
                    
                    if lat is not None and lon is not None:
                        try:
                            return (float(lat), float(lon))
                        except (ValueError, TypeError):
                            continue
                
                else:
                    key = key_info[0]
                    location = safe_get_value(metadata, key)
                    
                    if location is not None:
                        location_str = str(location)
                        
                        # Parse location string formats
                        if "+" in location_str:
                            parts = location_str.strip("/").split("+")
                            parts = [p for p in parts if p]
                            if len(parts) >= 2:
                                try:
                                    lat = float(parts[0])
                                    lon = float(parts[1])
                                    return (lat, lon)
                                except (ValueError, TypeError):
                                    continue
                        
                        for separator in [',', ' ', ';']:
                            if separator in location_str:
                                parts = location_str.split(separator)
                                if len(parts) >= 2:
                                    try:
                                        lat = float(parts[0].strip())
                                        lon = float(parts[1].strip())
                                        return (lat, lon)
                                    except (ValueError, TypeError):
                                        continue
            
            return None
            
    except Exception:
        return None


def get_video_datetime(file_path):
    """
    Extract creation datetime from video file metadata.
    
    Args:
        file_path: Path to video file
        
    Returns:
        datetime: Creation datetime or None if not found
    """
    try:
        parser = createParser(str(file_path))
        if not parser:
            return None
        metadata = extractMetadata(parser)
        if metadata and metadata.has("creation_date"):
            return metadata.get("creation_date")
    except Exception:
        pass
    return None