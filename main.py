#!/usr/bin/env python3
"""
Lunar Terrain Analysis Pipeline
===============================
A hybrid physics-AI system for detecting boulders and landslides on lunar terrain.
"""

import os
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime
from modules import utils
from modules.boulder_detection import HybridBoulderDetector
from modules.landslide_detection import HybridLandslideDetector
from modules.feedback_trainer import FeedbackTrainer

# Constants
DOWNSAMPLE_FACTOR = 4
PIXEL_RESOLUTION = 0.25  # meters per pixel (LRO NAC resolution)
MIN_BOULDER_AREA = 2  # square meters
MIN_LANDSLIDE_AREA = 10  # square meters

class LunarAnalysisPipeline:
    """Main pipeline for lunar terrain analysis."""
    
    def __init__(self):
        """Initialize the pipeline with proper logging and directories."""
        self._setup_logging()
        self._setup_directories()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pixel_resolution = PIXEL_RESOLUTION * DOWNSAMPLE_FACTOR  # Adjusted for downsampling
        
    def _setup_logging(self):
        """Configure logging system."""
        logging.basicConfig(
            filename="lunar_detection.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
    def _setup_directories(self):
        """Create necessary output directories."""
        Path("results").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        
    def run(self, image_path: str) -> bool:
        """Execute the full analysis pipeline.
        
        Args:
            image_path: Path to input TIFF image
            
        Returns:
            True if pipeline completed successfully
        """
        try:
            self.logger.info("🚀 Starting Lunar Terrain Analysis Pipeline")
            start_time = datetime.now()
            
            # Load and preprocess image
            image = self._load_and_preprocess(image_path)
            if image is None:
                return False
                
            # Terrain feature extraction
            slope, curvature, roughness = self._extract_terrain_features(image)
            
            # Boulder detection
            boulder_mask, boulder_stats = self._detect_boulders(image, slope)
            
            # Landslide detection
            landslide_mask, landslide_stats, sources = self._detect_landslides(
                image, slope, curvature, roughness
            )
            
            # Generate outputs
            self._generate_outputs(
                image, 
                boulder_mask, 
                landslide_mask, 
                sources,
                boulder_stats,
                landslide_stats
            )
            
            # Update feedback system
            self._update_feedback_system(boulder_mask, landslide_mask)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"✅ Pipeline completed in {duration:.2f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
            return False
            
    def _load_and_preprocess(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess input image."""
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found at {image_path}")
            return None
            
        try:
            self.logger.info(f"Loading image from {image_path}")
            full_image = utils.load_tiff(image_path)
            self.logger.info(f"Original image shape: {full_image.shape}")
            
            # Downsample for processing
            image = utils.auto_downsample(full_image, factor=DOWNSAMPLE_FACTOR)
            self.logger.info(f"Downsampled image shape: {image.shape}")
            
            # Normalize and convert to standard format
            image = self._standardize_image(image)
            return image
            
        except Exception as e:
            self.logger.error(f"Image loading failed: {str(e)}")
            return None
            
    def _standardize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to standard 8-bit format."""
        if image.dtype != np.uint8:
            image = (255 * utils.normalize(image)).astype(np.uint8)
        if image.ndim == 3 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        return image
        
    def _extract_terrain_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract terrain features from image."""
        self.logger.info("Extracting terrain features...")
        
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.squeeze()
            
        # Calculate slope (using Sobel derivatives)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        slope = np.arctan(np.sqrt(dx**2 + dy**2))  # in radians
        
        # Calculate curvature (Laplacian)
        curvature = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        
        # Calculate roughness (local standard deviation)
        roughness = cv2.blur(gray**2, (3,3)) - cv2.blur(gray, (3,3))**2
        roughness = np.sqrt(np.maximum(roughness, 0))
        
        return slope, curvature, roughness
        
    def _detect_boulders(self, image: np.ndarray, slope: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Detect boulders in the image."""
        self.logger.info("Detecting boulders...")
        
        detector = HybridBoulderDetector()
        mask, stats = detector.detect(image, terrain=slope, downsample=DOWNSAMPLE_FACTOR)
        
        # Convert area to square meters and filter small boulders
        filtered_stats = []
        for s in stats:
            # Ensure consistent statistics format
            if 'area' in s:  # From new detector version
                area_m2 = s['area']
            elif 'area_est' in s:  # From old detector version
                area_m2 = s['area_est']
            else:
                # Calculate area if not provided
                area_px = s.get('area_px', s.get('major_axis', 0) * s.get('minor_axis', 0))
                area_m2 = area_px * (self.pixel_resolution ** 2)
                
            if area_m2 >= MIN_BOULDER_AREA:
                # Standardize output format
                filtered_stats.append({
                    'centroid': s.get('centroid'),
                    'area_est': area_m2,
                    'radius': s.get('radius', np.sqrt(area_m2 / np.pi)),
                    'length_est': s.get('length_est', 2 * np.sqrt(area_m2 / np.pi))
                })
        
        self.logger.info(f"Detected {len(filtered_stats)} boulders")
        return mask, filtered_stats
        
    def _detect_landslides(self, 
                         image: np.ndarray,
                         slope: np.ndarray,
                         curvature: np.ndarray,
                         roughness: np.ndarray) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """Detect landslides in the image."""
        self.logger.info("Detecting landslides...")
        
        terrain = {
            'slope': slope,
            'curvature': curvature,
            'roughness': roughness
        }
        
        detector = HybridLandslideDetector()
        mask, stats, sources = detector.detect(image, terrain=terrain)
        
        # Filter small landslides and convert units
        filtered_stats = []
        for s in stats:
            if 'area' not in s:
                continue
                
            area_m2 = s['area'] * (self.pixel_resolution ** 2)
            if area_m2 >= MIN_LANDSLIDE_AREA:
                filtered_stats.append({
                    'area': area_m2,
                    'length': s.get('length', 0) * self.pixel_resolution,
                    'polygon': s.get('polygon')
                })
        
        self.logger.info(f"Detected {len(filtered_stats)} landslides")
        return mask, filtered_stats, sources
        
    def _generate_outputs(self,
                        image: np.ndarray,
                        boulder_mask: np.ndarray,
                        landslide_mask: np.ndarray,
                        sources: np.ndarray,
                        boulder_stats: List[Dict],
                        landslide_stats: List[Dict]):
        """Generate all output products."""
        self.logger.info("Generating output products...")
        
        # Save masks
        utils.save_tiff(boulder_mask, "results/boulder_mask.tif")
        utils.save_tiff(landslide_mask, "results/landslide_mask.tif")
        utils.save_tiff(sources, "results/landslide_sources.tif")
        
        # Create annotated visualization
        annotated_path = "results/annotated_output.png"
        utils.annotate_map(image, boulder_mask, landslide_mask, annotated_path)
        
        # Generate report
        self._generate_report(boulder_stats, landslide_stats)
        
    def _generate_report(self, boulder_stats: List[Dict], landslide_stats: List[Dict]):
        """Generate analysis report."""
        report_path = "results/analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Lunar Terrain Analysis Report\n")
            f.write("=============================\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Boulders detected: {len(boulder_stats)}\n")
            f.write(f"Landslides detected: {len(landslide_stats)}\n\n")
            
            f.write("Top 5 Largest Boulders:\n")
            for i, b in enumerate(sorted(boulder_stats, key=lambda x: -x['area_est'])[:5]):
                f.write(f"{i+1}. Area: {b['area_est']:.2f} m², "
                       f"Radius: {b['radius']:.2f} m\n")
                
            f.write("\nTop 3 Largest Landslides:\n")
            for i, l in enumerate(sorted(landslide_stats, key=lambda x: -x['area'])[:3]):
                f.write(f"{i+1}. Area: {l['area']:.2f} m², "
                       f"Length: {l['length']:.2f} m\n")
        
    def _update_feedback_system(self, boulder_mask: np.ndarray, landslide_mask: np.ndarray):
        """Update the feedback training system."""
        self.logger.info("Updating feedback system...")
        
        try:
            trainer = FeedbackTrainer(model_path="models/feedback_cnn.pt")
            
            # Generate training samples from detections
            patches = self._generate_training_patches(boulder_mask, landslide_mask)
            labels = [1] * len(patches)  # Placeholder - real system would have proper labels
            
            if len(patches) > 0:
                trainer.update_model(patches, labels, epochs=5)
            else:
                self.logger.warning("No training samples generated for feedback")
                
        except Exception as e:
            self.logger.warning(f"Feedback update skipped: {str(e)}")
            
    def _generate_training_patches(self, *masks: np.ndarray) -> List[np.ndarray]:
        """Generate training patches from detection masks."""
        # Placeholder implementation
        return []

if __name__ == "__main__":
    pipeline = LunarAnalysisPipeline()
    success = pipeline.run("data/stack.tif")
    exit(0 if success else 1)