import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Dict, Tuple, List, Optional
from skimage import measure
import logging


class HybridBoulderDetector:
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        """Initialize the boulder detector with optional CNN model.
        
        Args:
            model_path: Path to trained CNN model weights
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        if model_path:
            try:
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                self.logger.info(f"CNN model loaded from {model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load model. Using fallback thresholding. Error: {e}")
        else:
            self.logger.warning("No model path provided. Using fallback logic.")

    def detect(self, 
               image: np.ndarray, 
               terrain: Optional[np.ndarray] = None, 
               downsample: int = 1,
               min_area: int = 5) -> Tuple[np.ndarray, List[Dict]]:
        """Detect boulders in an image using hybrid approach.
        
        Args:
            image: Input image (H,W,3) or (H,W)
            terrain: Optional terrain data (slope, etc.)
            downsample: Downsample factor for size calculations
            min_area: Minimum area (in pixels) to consider as boulder
            
        Returns:
            Tuple of (mask, stats) where:
            - mask: Binary mask of detected boulders
            - stats: List of boulder properties dictionaries
        """
        # Convert to grayscale if needed
        gray = self._preprocess_image(image)
        norm = self._normalize(gray)
        binary = self._physics_threshold(norm)

        # Extract patches (boulder candidates)
        mask = np.zeros_like(gray, dtype=np.uint8)
        stats = []

        labels = measure.label(binary)
        props = measure.regionprops(labels)

        for prop in props:
            if prop.area < min_area:
                continue

            minr, minc, maxr, maxc = prop.bbox
            patch = norm[minr:maxr, minc:maxc]
            
            # Skip too small patches
            if min(patch.shape) < 8:
                continue

            # Classify using model or rules
            if self.model:
                is_boulder = self._classify_patch(patch)
            else:
                is_boulder = self._rule_based_classify(prop)

            if is_boulder:
                self._add_boulder_to_results(prop, mask, stats, downsample)

        return mask, stats

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if image.ndim == 3:
            if image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return image[:, :, 0]
        return image

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 uint8 range with robust percentiles."""
        p1, p99 = np.percentile(img, [1, 99])
        norm = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)
        return (norm * 255).astype(np.uint8)

    def _physics_threshold(self, img: np.ndarray) -> np.ndarray:
        """Apply physics-informed adaptive thresholding."""
        img = img.astype(np.uint8)  # Ensure correct type
        denoised = cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)
        enhanced = cv2.equalizeHist(denoised)
        return cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 3
        )

    def _rule_based_classify(self, prop) -> bool:
        """Rule-based boulder classification using shape metrics."""
        circ = self._circularity(prop)
        return circ > 0.4 and prop.eccentricity < 0.8

    def _circularity(self, prop) -> float:
        """Calculate circularity metric (1.0 = perfect circle)."""
        if prop.perimeter == 0:
            return 0
        return 4 * np.pi * prop.area / (prop.perimeter ** 2)

    def _classify_patch(self, patch: np.ndarray) -> bool:
        """Classify patch using CNN model with proper preprocessing."""
        with torch.no_grad():
            try:
                input_tensor = self.transform(patch).unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                prob = F.softmax(output, dim=1)[0, 1].item()
                return prob > 0.5
            except Exception as e:
                self.logger.error(f"Classification failed: {e}")
                return False

    def _add_boulder_to_results(self, prop, mask: np.ndarray, 
                              stats: List[Dict], downsample: int):
        """Add validated boulder to results mask and statistics."""
        # Create contour mask
        contour_img = np.zeros_like(mask, dtype=np.uint8)
        coords = prop.coords
        contour_img[coords[:, 0], coords[:, 1]] = 255
        mask = cv2.bitwise_or(mask, contour_img)

        # Calculate statistics
        stats.append({
            "centroid": prop.centroid,
            "area": prop.area * (downsample ** 2),
            "major_axis": prop.major_axis_length * downsample,
            "minor_axis": prop.minor_axis_length * downsample,
            "circularity": self._circularity(prop),
            "eccentricity": prop.eccentricity
        })