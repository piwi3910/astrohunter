import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars, EPSFBuilder
from photutils.background import Background2D, MedianBackground
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_dog
import tensorflow as tf
import logging
from typing import List, Dict, Optional, Tuple
import cv2

logger = logging.getLogger(__name__)

class DetailedAnalyzer:
    def __init__(self):
        self.psf_model = None
        self.ml_model = None
        self.min_confidence = 0.85

        # Load ML model if available
        try:
            self.ml_model = tf.keras.models.load_model('/app/models/asteroid_classifier.h5')
        except Exception as e:
            logger.warning(f"Could not load ML model: {str(e)}")

    def build_psf_model(self, image_data: np.ndarray, stars: List[Dict]) -> Optional[models.Gaussian2D]:
        """Build PSF model from reference stars"""
        try:
            # Extract star cutouts
            star_cutouts = []
            for star in stars:
                x, y = star['xcentroid'], star['ycentroid']
                cutout = image_data[int(y-5):int(y+5), int(x-5):int(x+5)]
                if cutout.shape == (10, 10):  # Ensure complete cutout
                    star_cutouts.append(cutout)

            if not star_cutouts:
                return None

            # Build empirical PSF
            epsf_builder = EPSFBuilder(oversampling=2, maxiters=3)
            epsf = epsf_builder(star_cutouts)

            # Fit 2D Gaussian to empirical PSF
            y, x = np.mgrid[:epsf.data.shape[0], :epsf.data.shape[1]]
            g_init = models.Gaussian2D()
            fit_g = fitting.LevMarLSQFitter()
            psf_model = fit_g(g_init, x, y, epsf.data)

            return psf_model

        except Exception as e:
            logger.error(f"Error building PSF model: {str(e)}")
            return None

    def extract_features(self, image_data: np.ndarray, x: int, y: int) -> Dict:
        """Extract features for a detection"""
        try:
            # Extract patch around detection
            patch_size = 16
            patch = image_data[y-patch_size:y+patch_size, x-patch_size:x+patch_size]
            if patch.shape != (32, 32):
                return {}

            # Calculate basic statistics
            mean = np.mean(patch)
            std = np.std(patch)
            max_val = np.max(patch)
            min_val = np.min(patch)

            # Calculate shape features
            gradient_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0)
            gradient_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Calculate moment features
            moments = cv2.moments(patch)
            hu_moments = cv2.HuMoments(moments)

            return {
                'intensity_stats': {
                    'mean': float(mean),
                    'std': float(std),
                    'max': float(max_val),
                    'min': float(min_val),
                    'dynamic_range': float(max_val - min_val)
                },
                'shape_features': {
                    'gradient_mean': float(np.mean(gradient_magnitude)),
                    'gradient_std': float(np.std(gradient_magnitude)),
                    'hu_moments': [float(m[0]) for m in hu_moments]
                }
            }

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}

    def verify_with_ml(self, image_data: np.ndarray, x: int, y: int) -> float:
        """Verify detection using ML model"""
        if self.ml_model is None:
            return 0.5  # Default confidence if no model

        try:
            # Extract patch and preprocess
            patch_size = 16
            patch = image_data[y-patch_size:y+patch_size, x-patch_size:x+patch_size]
            if patch.shape != (32, 32):
                return 0.0

            # Normalize patch
            patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
            
            # Reshape for model
            patch = np.expand_dims(np.expand_dims(patch, axis=-1), axis=0)

            # Get prediction
            prediction = self.ml_model.predict(patch, verbose=0)[0][0]
            return float(prediction)

        except Exception as e:
            logger.error(f"Error in ML verification: {str(e)}")
            return 0.0

    def analyze_trajectory(self, detections: List[Dict]) -> Dict:
        """Analyze object trajectory across multiple images"""
        if len(detections) < 2:
            return {}

        try:
            # Extract positions and times
            positions = np.array([(d['ra'], d['dec']) for d in detections])
            
            # Calculate velocity and acceleration
            velocity = np.diff(positions, axis=0)
            acceleration = np.diff(velocity, axis=0) if len(velocity) > 1 else np.zeros((1, 2))

            # Check for consistent motion
            velocity_consistency = np.std(velocity, axis=0)
            is_consistent = np.all(velocity_consistency < 0.1)  # Threshold for consistency

            return {
                'velocity_ra': float(np.mean(velocity[:, 0])),
                'velocity_dec': float(np.mean(velocity[:, 1])),
                'acceleration_ra': float(np.mean(acceleration[:, 0])),
                'acceleration_dec': float(np.mean(acceleration[:, 1])),
                'motion_consistency': float(is_consistent),
                'trajectory_length': float(np.sum(np.sqrt(np.sum(velocity**2, axis=1))))
            }

        except Exception as e:
            logger.error(f"Error analyzing trajectory: {str(e)}")
            return {}

    def detailed_analysis(self, fits_path: str, initial_detection: Dict) -> Optional[Dict]:
        """Perform detailed analysis of a detection"""
        try:
            with fits.open(fits_path) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                wcs = WCS(header)

                # Convert world coordinates to pixel coordinates
                pixels = wcs.world_to_pixel(
                    initial_detection['ra'],
                    initial_detection['dec']
                )
                x, y = int(pixels[0]), int(pixels[1])

                # Extract features
                features = self.extract_features(data, x, y)
                
                # ML verification
                ml_confidence = self.verify_with_ml(data, x, y)

                # PSF analysis
                psf_fit_quality = 0.0
                if self.psf_model:
                    try:
                        patch = data[y-5:y+5, x-5:x+5]
                        psf_fit = fitting.LevMarLSQFitter()(
                            self.psf_model, 
                            np.arange(10), 
                            np.arange(10), 
                            patch
                        )
                        psf_fit_quality = float(psf_fit.stds['amplitude'])
                    except Exception as e:
                        logger.warning(f"PSF fitting failed: {str(e)}")

                # Combine all evidence
                final_confidence = np.mean([
                    initial_detection['confidence'],
                    ml_confidence,
                    1.0 if psf_fit_quality > 0.7 else 0.0
                ])

                if final_confidence >= self.min_confidence:
                    return {
                        'ra': initial_detection['ra'],
                        'dec': initial_detection['dec'],
                        'magnitude': initial_detection['magnitude'],
                        'confidence': final_confidence,
                        'metadata': {
                            **features,
                            'ml_confidence': ml_confidence,
                            'psf_fit_quality': psf_fit_quality,
                            'initial_confidence': initial_detection['confidence']
                        }
                    }

                return None

        except Exception as e:
            logger.error(f"Error in detailed analysis: {str(e)}")
            return None