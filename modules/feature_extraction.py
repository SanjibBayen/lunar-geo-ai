import numpy as np
import cv2
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import gaussian_filter, laplace
from typing import Dict, Optional, Union

def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range with safety checks.
    
    Args:
        arr: Input array to normalize
        
    Returns:
        Normalized array in [0, 1] range
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy array")
        
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    if np.isnan(arr_min) or np.isnan(arr_max):
        raise ValueError("Array contains NaN values")
        
    if arr_max - arr_min < 1e-10:  # Handle constant arrays
        return np.zeros_like(arr)
        
    return (arr - arr_min) / (arr_max - arr_min)

def extract_slope(dem: np.ndarray) -> np.ndarray:
    """Estimate slope from DTM using Sobel gradient with edge padding.
    
    Args:
        dem: Digital elevation model (2D array)
        
    Returns:
        Normalized slope map
    """
    if dem.ndim != 2:
        raise ValueError("DEM must be 2D array")
        
    # Pad edges to reduce boundary artifacts
    dem_padded = cv2.copyMakeBorder(dem, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    dx = sobel(dem_padded, axis=1)[1:-1, 1:-1]
    dy = sobel(dem_padded, axis=0)[1:-1, 1:-1]
    slope = np.sqrt(dx**2 + dy**2)
    return normalize(slope)

def extract_curvature(dem: np.ndarray) -> np.ndarray:
    """Estimate curvature (Laplacian of elevation) with edge padding.
    
    Args:
        dem: Digital elevation model (2D array)
        
    Returns:
        Normalized curvature map
    """
    if dem.ndim != 2:
        raise ValueError("DEM must be 2D array")
        
    # Pad edges to reduce boundary artifacts
    dem_padded = cv2.copyMakeBorder(dem, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    curv = laplace(dem_padded)[1:-1, 1:-1]
    return normalize(curv)

def extract_texture_features(patch: np.ndarray) -> Dict[str, float]:
    """Compute GLCM texture features for a grayscale patch with validation.
    
    Args:
        patch: Input image patch (2D array)
        
    Returns:
        Dictionary of texture features
    """
    if patch.ndim != 2:
        raise ValueError("Patch must be 2D grayscale")
        
    if patch.size < 4:  # Minimum size for GLCM
        return {
            'contrast': 0.0,
            'homogeneity': 1.0,
            'energy': 0.0,
            'correlation': 0.0
        }
        
    # Convert to 8-bit with auto-scaling
    patch_norm = normalize(patch)
    patch_8bit = (patch_norm * 255).astype(np.uint8)
    
    try:
        glcm = graycomatrix(
            patch_8bit, 
            distances=[1], 
            angles=[0], 
            levels=256,
            symmetric=True, 
            normed=True
        )
        return {
            'contrast': graycoprops(glcm, 'contrast')[0, 0],
            'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
            'energy': graycoprops(glcm, 'energy')[0, 0],
            'correlation': graycoprops(glcm, 'correlation')[0, 0]
        }
    except Exception:
        # Fallback if GLCM fails
        return {
            'contrast': 0.0,
            'homogeneity': 1.0,
            'energy': 0.0,
            'correlation': 0.0
        }

def extract_edge_features(image: np.ndarray) -> np.ndarray:
    """Extract edge and gradient strength from image with validation.
    
    Args:
        image: Input image (2D array)
        
    Returns:
        Normalized edge magnitude map
    """
    if image.ndim != 2:
        raise ValueError("Image must be 2D grayscale")
        
    # Pad edges to reduce boundary artifacts
    padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    sobelx = sobel(padded, axis=1)[1:-1, 1:-1]
    sobely = sobel(padded, axis=0)[1:-1, 1:-1]
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return normalize(magnitude)

def extract_shadow_mask(image: np.ndarray, threshold: int = 20) -> np.ndarray:
    """Detect shadows (dark regions) with adaptive thresholding.
    
    Args:
        image: Input image (2D array)
        threshold: Intensity threshold for shadows
        
    Returns:
        Normalized shadow mask [0,1]
    """
    if image.ndim != 2:
        raise ValueError("Image must be 2D grayscale")
        
    if image.dtype != np.uint8:
        image_norm = normalize(image)
        image_8bit = (image_norm * 255).astype(np.uint8)
    else:
        image_8bit = image
        
    blurred = cv2.GaussianBlur(image_8bit, (5, 5), 0)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask.astype(float) / 255.0

def extract_local_variance(patch: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Compute local variance with Gaussian smoothing.
    
    Args:
        patch: Input image patch (2D array)
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Normalized local variance map
    """
    if patch.ndim != 2:
        raise ValueError("Patch must be 2D array")
        
    patch_float = patch.astype(np.float32)
    blurred = gaussian_filter(patch_float, sigma=sigma)
    variance = (patch_float - blurred) ** 2
    return normalize(variance)

def extract_features(
    patch: np.ndarray, 
    dem_patch: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Extract comprehensive features from an image patch with optional terrain data.
    
    Args:
        patch: Input image patch (2D array)
        dem_patch: Optional DEM patch for terrain features (2D array)
        
    Returns:
        Dictionary of extracted features
    """
    if patch.ndim != 2:
        raise ValueError("Image patch must be 2D grayscale")
        
    features: Dict[str, Union[np.ndarray, float]] = {}
    
    # Base image features
    features['edges'] = extract_edge_features(patch)
    features['variance'] = extract_local_variance(patch)
    features['shadow'] = extract_shadow_mask(patch)
    
    # Texture features
    texture = extract_texture_features(patch)
    features.update(texture)

    # Terrain features if available
    if dem_patch is not None:
        if dem_patch.ndim != 2:
            raise ValueError("DEM patch must be 2D array")
        if dem_patch.shape != patch.shape:
            raise ValueError("DEM patch must match image dimensions")
            
        features['slope'] = extract_slope(dem_patch)
        features['curvature'] = extract_curvature(dem_patch)

    return features