import numpy as np
from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import logging
from typing import List, Dict, Optional
import cv2

logger = logging.getLogger(__name__)

class FastDetector:
    def __init__(self):
        self.threshold_sigma = 3.0  # Detection threshold in sigma
        self.fwhm = 3.0  # Full width at half maximum
        self.min_separation = 2.0  # Minimum separation between sources

    def preprocess_image(self, data: np.ndarray) -> np.ndarray:
        """Preprocess the image data"""
        try:
            # Convert to float32 if needed
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            # Basic noise reduction
            denoised = cv2.fastNlMeansDenoising(
                data.astype(np.uint8),
                None,
                h=10,
                searchWindowSize=21,
                templateWindowSize=7
            )

            # Enhance contrast
            normalized = cv2.normalize(
                denoised,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_32F
            )

            return normalized

        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            return data

    def detect_sources(self, fits_path: str) -> List[Dict]:
        """Detect sources in a FITS image"""
        try:
            # Open FITS file
            with fits.open(fits_path) as hdul:
                # Get the primary data
                data = hdul[0].data
                header = hdul[0].header
                wcs = WCS(header)

                # Preprocess the image
                processed_data = self.preprocess_image(data)

                # Calculate background statistics
                mean, median, std = sigma_clipped_stats(processed_data)

                # Create DAOStarFinder object
                daofind = DAOStarFinder(
                    fwhm=self.fwhm,
                    threshold=self.threshold_sigma * std,
                    sharplo=0.2,
                    sharphi=1.0,
                    roundlo=-0.5,
                    roundhi=0.5,
                )

                # Find sources
                sources = daofind(processed_data - median)

                if sources is None:
                    return []

                detections = []
                for source in sources:
                    try:
                        # Convert pixel coordinates to world coordinates
                        ra_dec = wcs.pixel_to_world(
                            source['xcentroid'],
                            source['ycentroid']
                        )

                        detection = {
                            'ra': float(ra_dec.ra.deg),
                            'dec': float(ra_dec.dec.deg),
                            'magnitude': -2.5 * np.log10(source['flux']),
                            'confidence': float(source['peak'] / std),
                            'metadata': {
                                'sharpness': float(source['sharpness']),
                                'roundness': float(source['roundness1']),
                                'peak_value': float(source['peak']),
                                'flux': float(source['flux'])
                            }
                        }
                        detections.append(detection)

                    except Exception as e:
                        logger.warning(f"Error processing source: {str(e)}")
                        continue

                return detections

        except Exception as e:
            logger.error(f"Error in source detection: {str(e)}")
            return []

    def filter_detections(self, detections: List[Dict], min_confidence: float = 0.8) -> List[Dict]:
        """Filter detections based on confidence and other criteria"""
        filtered = []
        for detection in detections:
            if (detection['confidence'] >= min_confidence and
                detection['metadata']['sharpness'] > 0.2 and
                abs(detection['metadata']['roundness']) < 0.5):
                filtered.append(detection)
        return filtered

    def analyze_image_pair(self, image1_path: str, image2_path: str) -> List[Dict]:
        """Analyze a pair of images for moving objects"""
        # Detect sources in both images
        detections1 = self.detect_sources(image1_path)
        detections2 = self.detect_sources(image2_path)

        # Filter detections
        detections1 = self.filter_detections(detections1)
        detections2 = self.filter_detections(detections2)

        # Find potential moving objects by comparing positions
        moving_objects = []
        for d1 in detections1:
            for d2 in detections2:
                # Calculate angular separation
                ra_diff = abs(d1['ra'] - d2['ra'])
                dec_diff = abs(d1['dec'] - d2['dec'])
                separation = np.sqrt(ra_diff**2 + dec_diff**2)

                # If objects are close but not identical, might be moving
                if 0.0001 < separation < 0.001:  # Adjust these thresholds as needed
                    moving_objects.append({
                        'ra': d2['ra'],
                        'dec': d2['dec'],
                        'magnitude': d2['magnitude'],
                        'confidence': min(d1['confidence'], d2['confidence']),
                        'metadata': {
                            'movement': separation * 3600,  # Convert to arcseconds
                            'initial_position': {'ra': d1['ra'], 'dec': d1['dec']},
                            'final_position': {'ra': d2['ra'], 'dec': d2['dec']},
                            'magnitude_change': d2['magnitude'] - d1['magnitude']
                        }
                    })

        return moving_objects