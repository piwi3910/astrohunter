"""
Module for fast initial detection of moving objects.
"""
import numpy as np
from astropy.io import fits
from skimage import registration
from photutils.detection import DAOStarFinder
from dataclasses import dataclass
import logging
from typing import List, Tuple, Optional
from astropy.stats import sigma_clipped_stats

logger = logging.getLogger('astrohunter')

@dataclass
class Candidate:
    """Class for storing candidate moving object information."""
    x: float  # X position in first image
    y: float  # Y position in first image
    movement_px: float  # Movement in pixels
    direction: float  # Direction in degrees
    snr: float  # Signal to noise ratio
    time_diff: float  # Time difference in hours

class FastDetector:
    def __init__(self,
                 snr_threshold: float = 4.0,
                 min_movement: float = 0.3,  # pixels
                 max_movement: float = 40.0):  # pixels
        """
        Initialize the fast detector.
        
        Args:
            snr_threshold: Signal to noise ratio threshold
            min_movement: Minimum movement in pixels
            max_movement: Maximum movement in pixels
        """
        self.snr_threshold = snr_threshold
        self.min_movement = min_movement
        self.max_movement = max_movement
        
    def read_fits(self, filename: str) -> Optional[np.ndarray]:
        """
        Read and preprocess FITS image.
        
        Args:
            filename: Path to FITS file
            
        Returns:
            Preprocessed image data or None if invalid
        """
        try:
            with fits.open(filename) as hdul:
                # Check if we have valid data
                if len(hdul) == 0 or hdul[0].data is None:
                    logger.warning(f"No valid data in {filename}")
                    return None
                    
                # Get data and ensure it's 2D
                data = hdul[0].data
                
                # If data is 3D or higher, take the first 2D slice
                while data.ndim > 2:
                    data = data[0]
                
                data = data.astype(np.float32)
                
                # Basic validation
                if data.ndim != 2 or data.size == 0 or not np.any(np.isfinite(data)):
                    logger.warning(f"No valid pixels in {filename}")
                    return None
                    
                # Remove any NaN or infinite values
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Calculate robust statistics
                mean, median, std = sigma_clipped_stats(data, sigma=3.0)
                
                if std == 0:
                    logger.warning(f"Zero standard deviation in {filename}")
                    return None
                    
                # Scale data to reasonable range for both source detection and PSF analysis
                data = data - median
                if std > 0:
                    data = data / std
                data = np.clip(data, 0, 50)  # Allow higher dynamic range
                
                return data
                
        except Exception as e:
            logger.error(f"Error reading {filename}: {str(e)}")
            return None
        
    def align_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two images using phase correlation.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Tuple of aligned images
        """
        try:
            # Ensure images are not empty and have valid values
            if img1 is None or img2 is None:
                return None, None
                
            if not np.any(img1) or not np.any(img2):
                return img1, img2
                
            # Calculate shift between images
            shift, error, _ = registration.phase_cross_correlation(img1, img2)
            
            if np.any(np.isnan(shift)):
                return img1, img2
                
            # Apply shift to second image
            img2_aligned = np.roll(np.roll(img2, int(shift[0]), axis=0), int(shift[1]), axis=1)
            
            return img1, img2_aligned
            
        except Exception as e:
            logger.error(f"Error aligning images: {str(e)}")
            return img1, img2
        
    def find_sources(self, image: np.ndarray, threshold: float = 3.0) -> List[Tuple[float, float]]:
        """
        Find sources in image using DAOStarFinder.
        
        Args:
            image: Image data
            threshold: Detection threshold in sigma
            
        Returns:
            List of (x, y) coordinates
        """
        try:
            if image is None:
                return []
                
            # Calculate background statistics
            mean, median, std = sigma_clipped_stats(image, sigma=3.0)
            
            if std == 0:
                return []
                
            # Find sources with more sensitive parameters
            daofind = DAOStarFinder(fwhm=2.0, threshold=2.0*std, sharplo=0.2, sharphi=1.0)
            sources = daofind(image)  # Image already background-subtracted
            
            if sources is None:
                return []
                
            return [(x, y) for x, y in zip(sources['xcentroid'], sources['ycentroid'])]
            
        except Exception as e:
            logger.error(f"Error finding sources: {str(e)}")
            return []
        
    def match_sources(self,
                     sources1: List[Tuple[float, float]],
                     sources2: List[Tuple[float, float]],
                     max_dist: float = 40.0) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Match sources between two lists within maximum distance.
        
        Args:
            sources1: List of (x, y) coordinates from first image
            sources2: List of (x, y) coordinates from second image
            max_dist: Maximum matching distance in pixels
            
        Returns:
            List of matched source pairs
        """
        matches = []
        
        for x1, y1 in sources1:
            for x2, y2 in sources2:
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if dist <= max_dist:
                    matches.append(((x1, y1), (x2, y2)))
                    
        return matches
        
    def find_candidates(self,
                       file1: str,
                       file2: str,
                       time_diff: float) -> List[Candidate]:
        """
        Find moving object candidates between two images.
        
        Args:
            file1: Path to first FITS file
            file2: Path to second FITS file
            time_diff: Time difference in hours
            
        Returns:
            List of candidates
        """
        try:
            # Read images
            img1 = self.read_fits(file1)
            img2 = self.read_fits(file2)
            
            if img1 is None or img2 is None:
                return []
                
            # Align images
            img1, img2 = self.align_images(img1, img2)
            
            if img1 is None or img2 is None:
                return []
                
            # Find sources in both images
            sources1 = self.find_sources(img1)
            sources2 = self.find_sources(img2)
            
            if not sources1 or not sources2:
                return []
                
            # Match sources
            matches = self.match_sources(sources1, sources2, self.max_movement)
            
            candidates = []
            for (x1, y1), (x2, y2) in matches:
                # Calculate movement
                movement = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if movement < self.min_movement or movement > self.max_movement:
                    continue
                    
                # Calculate direction
                direction = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # Calculate SNR (use average of both positions)
                snr1 = self.calculate_snr(img1, x1, y1)
                snr2 = self.calculate_snr(img2, x2, y2)
                snr = (snr1 + snr2) / 2
                
                if snr >= self.snr_threshold:
                    candidates.append(Candidate(
                        x=x1,
                        y=y1,
                        movement_px=movement,
                        direction=direction,
                        snr=snr,
                        time_diff=time_diff
                    ))
                    
            return candidates
            
        except Exception as e:
            logger.error(f"Error finding candidates: {str(e)}")
            return []
            
    def calculate_snr(self, image: np.ndarray, x: float, y: float, radius: int = 5) -> float:
        """
        Calculate signal to noise ratio for a source.
        
        Args:
            image: Image data
            x: X coordinate
            y: Y coordinate
            radius: Radius for SNR calculation
            
        Returns:
            SNR value
        """
        try:
            if image is None:
                return 0.0
                
            # Extract region around source
            x_int, y_int = int(x), int(y)
            region = image[
                max(0, y_int-radius):min(image.shape[0], y_int+radius+1),
                max(0, x_int-radius):min(image.shape[1], x_int+radius+1)
            ]
            
            if region.size == 0:
                return 0.0
                
            # Calculate robust statistics
            mean, median, std = sigma_clipped_stats(region, sigma=3.0)
            
            if std == 0:
                return 0.0
                
            return (mean - median) / std
            
        except Exception as e:
            logger.error(f"Error calculating SNR: {str(e)}")
            return 0.0