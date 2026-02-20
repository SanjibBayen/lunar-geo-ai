import numpy as np
import cv2
import logging
from typing import Dict, Tuple, List, Optional
from skimage import measure
import torch
import torch.nn.functional as F
from torchvision import transforms


class HybridLandslideDetector:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the HybridLandslideDetector.
        
        Args:
            model_path: Path to trained PyTorch model for landslide detection
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        if model_path:
            try:
                self.model = torch.load(model_path, map_location='cpu')
                self.model.eval()
                self.logger.info(f"CNN landslide model loaded from {model_path}")
            except Exception as e:
                self.logger.warning(f"Model load failed. Using hybrid fallback. ({e})")
        else:
            self.logger.warning("No ML model provided. Using hybrid fallback.")

    def detect(self, 
               image: np.ndarray, 
               terrain: Optional[Dict[str, np.ndarray]] = None, 
               downsample: int = 1) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """Detect landslides in an image using hybrid physics-AI approach.
        
        Args:
            image: Input image (H,W,3) or (H,W)
            terrain: Dictionary containing terrain features:
                - slope: Slope map (required)
                - curvature: Curvature map (required)
                - roughness: Roughness map (optional)
            downsample: Downsample factor for size calculations
            
        Returns:
            Tuple containing:
            - mask: Binary mask of detected landslides (255=landslide, 0=background)
            - stats: List of landslide properties dictionaries
            - sources: Source points map (255=source point, 0=background)
        """
        # Validate input image
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")
        if image.ndim not in (2, 3):
            raise ValueError("Input image must be 2D grayscale or 3D RGB")

        # Handle missing terrain data
        if terrain is None:
            self.logger.warning("No terrain data provided. Using zeros.")
            terrain = {
                'slope': np.zeros_like(image) if image.ndim == 2 else np.zeros_like(image[:, :, 0]),
                'curvature': np.zeros_like(image) if image.ndim == 2 else np.zeros_like(image[:, :, 0]),
                'roughness': np.zeros_like(image) if image.ndim == 2 else np.zeros_like(image[:, :, 0])
            }

        # Validate terrain data
        required_keys = {'slope', 'curvature'}
        if not all(k in terrain for k in required_keys):
            missing = required_keys - set(terrain.keys())
            raise ValueError(f"Missing required terrain data: {missing}")

        slope = terrain['slope']
        curvature = terrain['curvature']
        roughness = terrain.get('roughness', np.zeros_like(slope))

        # Validate terrain shapes
        if slope.shape != curvature.shape:
            raise ValueError("Slope and curvature maps must have same dimensions")
        if roughness.shape != slope.shape:
            raise ValueError("Roughness map must match slope dimensions")

        try:
            # Normalize features with safety checks
            slope_n = self._normalize_feature(slope, 'slope')
            curv_n = self._normalize_feature(curvature, 'curvature')
            rough_n = self._normalize_feature(roughness, 'roughness')

            # Hybrid physics-informed score
            landslide_score = (
                0.5 * slope_n +
                0.3 * curv_n +
                0.2 * rough_n
            )

            # Threshold-based mask
            score_mask = (landslide_score > 0.6).astype(np.uint8) * 255

            # Optional ML refinement
            if self.model:
                score_mask = self._ml_refine(score_mask, slope_n, curv_n)

            # Process detected regions
            mask, stats, sources = self._process_regions(score_mask, slope, downsample)

            return mask, stats, sources

        except Exception as e:
            self.logger.error(f"Landslide detection failed: {e}")
            raise

    def _normalize_feature(self, feature: np.ndarray, name: str) -> np.ndarray:
        """Normalize terrain feature to [0,1] range with safety checks."""
        if not isinstance(feature, np.ndarray):
            raise ValueError(f"{name} must be a numpy array")
        
        if np.all(feature == 0):  # Handle case where feature is all zeros
            return np.zeros_like(feature, dtype=np.float32)
        
        feature_min = np.min(feature)
        feature_max = np.max(feature)
        
        if feature_max - feature_min < 1e-6:  # Handle constant-valued features
            return np.zeros_like(feature, dtype=np.float32)
            
        return (feature - feature_min) / (feature_max - feature_min)

    def _process_regions(self, 
                        score_mask: np.ndarray, 
                        slope: np.ndarray,
                        downsample: int) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """Process candidate regions and extract landslide properties."""
        mask = np.zeros_like(score_mask, dtype=np.uint8)
        stats = []
        sources = np.zeros_like(mask, dtype=np.uint8)

        # Find connected components
        labels = measure.label(score_mask > 0)
        props = measure.regionprops(labels)

        for prop in props:
            if prop.area < 10:  # Skip small regions
                continue

            # Classify region
            if not self._rule_or_cnn_classify(prop):
                continue

            # Add to mask
            coords = prop.coords
            mask[coords[:, 0], coords[:, 1]] = 255

            # Estimate source point
            source_point = self._estimate_source(prop, slope)
            sources[source_point] = 255

            # Record statistics
            stats.append({
                "area": prop.area * (downsample ** 2),
                "bbox": prop.bbox,
                "centroid": prop.centroid,
                "source": source_point,
                "perimeter": prop.perimeter * downsample,
                "circularity": 4 * np.pi * prop.area / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
            })

        return mask, stats, sources

    def _ml_refine(self, 
                  mask: np.ndarray, 
                  slope: np.ndarray, 
                  curvature: np.ndarray) -> np.ndarray:
        """Refine detection using ML model.
        
        Args:
            mask: Initial detection mask
            slope: Normalized slope map
            curvature: Normalized curvature map
            
        Returns:
            Refined detection mask
        """
        if self.model is None:
            return mask

        try:
            # Convert to PyTorch tensors
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            # Process image patches (placeholder implementation)
            # In practice, you would extract patches and run through model
            return mask
            
        except Exception as e:
            self.logger.warning(f"ML refinement failed: {e}")
            return mask

    def _rule_or_cnn_classify(self, region) -> bool:
        """Classify region using rules or CNN.
        
        Args:
            region: skimage.measure._regionprops.RegionProperties object
            
        Returns:
            True if region is classified as landslide
        """
        # Basic shape filtering
        circularity = 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0
        elongation = region.major_axis_length / (region.minor_axis_length + 1e-6)
        
        # Simple rule-based classifier
        return circularity < 0.3 and elongation > 1.5

    def _estimate_source(self, 
                        region, 
                        slope_img: np.ndarray) -> Tuple[int, int]:
        """Estimate landslide source point (point with maximum slope).
        
        Args:
            region: skimage.measure._regionprops.RegionProperties object
            slope_img: Slope map
            
        Returns:
            (row, col) coordinates of estimated source point
        """
        coords = region.coords
        if len(coords) == 0:
            return (0, 0)
        
        # Find point with maximum slope in the region
        slopes = slope_img[coords[:, 0], coords[:, 1]]
        max_idx = np.argmax(slopes)
        return tuple(coords[max_idx])