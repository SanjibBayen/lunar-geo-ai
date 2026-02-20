"""
Utility functions for Lunar Terrain Analysis Pipeline
====================================================
Image processing and file I/O operations for lunar terrain analysis.
"""

import os
import cv2
import numpy as np
import tifffile
import rasterio
from skimage.util import view_as_windows
from typing import Tuple, List, Dict, Union, Optional
from pathlib import Path
from datetime import datetime

def load_tiff(path: str, 
              bands: Tuple[int, int, int] = (1, 2, 3), 
              downsample_factor: int = 4) -> np.ndarray:
    """
    Load and downsample multi-band TIFF image.
    
    Args:
        path: Path to TIFF file
        bands: Band indices to load (1-based)
        downsample_factor: Downsampling factor
        
    Returns:
        Downsampled image array (H, W, C)
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: For invalid parameters
        rasterio.RasterioIOError: For file reading errors
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    if downsample_factor < 1:
        raise ValueError("Downsample factor must be ≥1")
    
    try:
        with rasterio.open(path) as src:
            # Validate bands
            if any(b < 1 or b > src.count for b in bands):
                raise ValueError(f"Band indices must be between 1 and {src.count}")
                
            downsampled_bands = []
            for b in bands:
                band = src.read(b, out_dtype='float32')
                h, w = band.shape
                
                # Calculate new dimensions
                new_w = max(1, w // downsample_factor)
                new_h = max(1, h // downsample_factor)
                
                band_ds = cv2.resize(band, (new_w, new_h), 
                                    interpolation=cv2.INTER_AREA)
                downsampled_bands.append(band_ds)
                
            return np.stack(downsampled_bands, axis=-1)
            
    except Exception as e:
        raise rasterio.RasterioIOError(f"Error reading {path}: {str(e)}")

def save_tiff(image: np.ndarray, path: str) -> None:
    """
    Save image as TIFF with proper compression.
    
    Args:
        image: 2D numpy array to save
        path: Output file path
        
    Raises:
        ValueError: If image is not 2D
        IOError: If file cannot be written
    """
    if image.ndim != 2:
        raise ValueError("Input image must be single-band (2D array)")
    
    try:
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Try different compression methods with fallbacks
        compression_methods = ['lzw', 'deflate', 'packbits', None]
        
        for method in compression_methods:
            try:
                if method is None:
                    tifffile.imwrite(path, image.astype(np.float32))
                    break
                else:
                    tifffile.imwrite(
                        path, 
                        image.astype(np.float32), 
                        compression=method
                    )
                    break
            except Exception as e:
                if method == compression_methods[-1]:  # Last method failed
                    raise IOError(f"All compression methods failed: {str(e)}")
                continue
                
    except Exception as e:
        raise IOError(f"Failed to save TIFF file {path}: {str(e)}")

def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 1] range with safe division.
    
    Args:
        arr: Input array
        
    Returns:
        Normalized array
    """
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    return (arr - arr_min) / (arr_max - arr_min + np.finfo(float).eps)

def auto_downsample(image: np.ndarray, 
                   factor: int = 4) -> np.ndarray:
    """
    Downsample image by specified factor.
    
    Args:
        image: Input image (2D or 3D)
        factor: Downsampling factor
        
    Returns:
        Downsampled image
        
    Raises:
        ValueError: For invalid factor or image size
    """
    if factor < 1:
        raise ValueError("Downsampling factor must be ≥1")
        
    h, w = image.shape[:2]
    if h < factor or w < factor:
        raise ValueError(f"Downsampling factor {factor} too large for image size ({h}, {w})")

    new_size = (max(1, w // factor), max(1, h // factor))
    
    if image.ndim == 3:
        bands = image.shape[2]
        resized = np.zeros((new_size[1], new_size[0], bands), dtype=image.dtype)
        for i in range(bands):
            resized[:, :, i] = cv2.resize(
                image[:, :, i], new_size, interpolation=cv2.INTER_AREA
            )
        return resized
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def tile_image(image: np.ndarray, 
              tile_size: int = 512, 
              stride: Optional[int] = None) -> np.ndarray:
    """
    Split image into overlapping tiles.
    
    Args:
        image: 2D input image
        tile_size: Size of square tiles
        stride: Step between tiles
        
    Returns:
        Array of tiles (num_tiles, tile_size, tile_size)
    """
    stride = stride or tile_size
    if image.ndim != 2:
        raise ValueError("Input image must be 2D")
        
    # Pad image if needed
    h, w = image.shape
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    tiles = view_as_windows(image, (tile_size, tile_size), step=stride)
    return tiles.reshape(-1, tile_size, tile_size)

def reconstruct_from_tiles(tiles: np.ndarray, 
                         image_shape: Tuple[int, int], 
                         tile_size: int = 512) -> np.ndarray:
    """
    Reconstruct image from non-overlapping tiles.
    
    Args:
        tiles: Array of tiles
        image_shape: Original image dimensions
        tile_size: Size of tiles
        
    Returns:
        Reconstructed image
    """
    h, w = image_shape
    output = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    idx = 0
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            if i + tile_size <= h and j + tile_size <= w:
                output[i:i+tile_size, j:j+tile_size] += tiles[idx]
                count[i:i+tile_size, j:j+tile_size] += 1
                idx += 1

    return np.divide(output, count, out=np.zeros_like(output), where=count != 0)

def draw_mask_on_image(base_image: np.ndarray, 
                      mask: np.ndarray, 
                      color: Tuple[int, int, int] = (0, 255, 0), 
                      alpha: float = 0.4) -> np.ndarray:
    """
    Overlay mask on base image with transparency.
    
    Args:
        base_image: Background image
        mask: Binary mask to overlay
        color: RGB color for mask
        alpha: Transparency level
        
    Returns:
        Blended BGR image
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D array")
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # Convert base image to 3-channel BGR
    if base_image.ndim == 2:
        base_image = cv2.cvtColor(
            (normalize(base_image) * 255).astype(np.uint8), 
            cv2.COLOR_GRAY2BGR
        )
    elif base_image.ndim == 3 and base_image.shape[2] == 1:
        base_image = cv2.cvtColor(
            (normalize(base_image[:, :, 0]) * 255).astype(np.uint8), 
            cv2.COLOR_GRAY2BGR
        )
    elif base_image.ndim == 3:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)

    # Create overlay
    overlay = base_image.copy()
    colored_mask = np.zeros_like(overlay)
    colored_mask[mask > 0] = color
    
    return cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)

def annotate_map(base_image: np.ndarray, 
                boulder_mask: np.ndarray, 
                landslide_mask: np.ndarray, 
                output_path: str) -> None:
    """
    Save annotated map with detections.
    
    Args:
        base_image: Background image
        boulder_mask: Boulder detections
        landslide_mask: Landslide detections
        output_path: Output file path
    """
    try:
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Overlay masks
        annotated = draw_mask_on_image(
            base_image, landslide_mask, color=(0, 0, 255), alpha=0.5
        )
        annotated = draw_mask_on_image(
            annotated, boulder_mask, color=(0, 255, 0), alpha=0.5
        )

        # Downsample if too large
        if annotated.shape[0] > 2000:
            scale = 2000 / annotated.shape[0]
            annotated = cv2.resize(
                annotated, 
                (int(annotated.shape[1] * scale), int(annotated.shape[0] * scale))
            )

        # Add timestamp
        cv2.putText(
            annotated,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        
        # Save as JPEG
        cv2.imwrite(output_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
    except Exception as e:
        raise RuntimeError(f"Failed to save annotated map: {str(e)}")

def estimate_boulder_size(mask: np.ndarray, 
                         pixel_resolution: float) -> List[Dict[str, Union[float, Tuple[int, int]]]]:
    """
    Calculate boulder statistics from mask.
    
    Args:
        mask: Binary detection mask
        pixel_resolution: Meters per pixel
        
    Returns:
        List of boulder statistics dictionaries
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    stats = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2:  # Skip small artifacts
            continue
            
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        length = np.sqrt(area) * pixel_resolution
        
        stats.append({
            'centroid': (int(x), int(y)),
            'radius': radius * pixel_resolution,
            'length_est': length,
            'area_est': area * (pixel_resolution ** 2)
        })
        
    return stats