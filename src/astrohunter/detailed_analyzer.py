"""
Module for detailed analysis of asteroid candidates.
"""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mpc import MPC
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .fast_detector import Candidate
from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder
from photutils.background import Background2D, MedianBackground
import logging

logger = logging.getLogger('astrohunter')

@dataclass
class DetailedCandidate:
    """Detailed candidate information after thorough analysis."""
    initial_candidate: Candidate
    ra: float  # Right ascension in degrees
    dec: float  # Declination in degrees
    proper_motion: float  # Proper motion in arcsec/hour
    position_angle: float  # Position angle in degrees
    psf_quality: float  # PSF quality metric (0-1)
    known_object: bool  # Whether it matches a known object
    magnitude: float  # Estimated magnitude
    confidence_score: float  # Overall confidence score (0-1)

class DetailedAnalyzer:
    def __init__(self,
                 psf_size: int = 25,
                 match_radius: float = 30.0,  # arcseconds
                 min_confidence: float = 0.7):
        """
        Initialize the detailed analyzer.
        
        Args:
            psf_size: Size of PSF stamp in pixels
            match_radius: Radius for matching with known objects (arcsec)
            min_confidence: Minimum confidence score to consider valid
        """
        self.psf_size = psf_size
        self.match_radius = match_radius
        self.min_confidence = min_confidence

    def get_wcs(self, fits_path: str) -> Optional[WCS]:
        """
        Get WCS information from FITS file.
        
        Args:
            fits_path: Path to FITS file
            
        Returns:
            WCS object or None if invalid
        """
        try:
            with fits.open(fits_path) as hdul:
                if len(hdul) == 0 or hdul[0].header is None:
                    return None
                wcs = WCS(hdul[0].header)
                if not wcs.has_celestial:
                    return None
                return wcs
        except Exception as e:
            logger.error(f"Error getting WCS: {str(e)}")
            return None

    def pixel_to_world(self, x: float, y: float, wcs: WCS) -> Optional[SkyCoord]:
        """
        Convert pixel coordinates to world coordinates.
        
        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            wcs: WCS object
            
        Returns:
            SkyCoord object or None if invalid
        """
        try:
            if wcs is None:
                return None
            return SkyCoord.from_pixel(x, y, wcs)
        except Exception as e:
            logger.error(f"Error converting coordinates: {str(e)}")
            return None

    def check_known_objects(self, coord: SkyCoord, time: str) -> bool:
        """
        Check if position matches any known objects.
        
        Args:
            coord: Sky coordinates to check
            time: Time of observation
            
        Returns:
            True if matches known object
        """
        try:
            # Query Minor Planet Center using newer API
            table = MPC.get_ephemeris(
                target='asteroid',
                location='500',  # Pan-STARRS observatory code
                epoch=time,
                ra=coord.ra.deg,
                dec=coord.dec.deg,
                radius=self.match_radius * u.arcsec
            )
            
            return len(table) > 0
            
        except Exception as e:
            logger.error(f"Error checking known objects: {str(e)}")
            return False

    def analyze_psf(self, image: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        """
        Analyze PSF quality and estimate magnitude.
        
        Args:
            image: Image data
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Tuple of (PSF quality score, estimated magnitude)
        """
        try:
            if image is None:
                return 0.0, 0.0
                
            # Extract PSF stars with more sensitive parameters
            stars_tbl = extract_stars(
                image,
                catalogs=[(x, y)],
                size=self.psf_size,
                min_separation=5.0,  # Allow closer stars
                edge_cutoff=3  # Smaller edge cutoff
            )
            
            if len(stars_tbl) == 0:
                return 0.0, 0.0
                
            # Build PSF model
            epsf_builder = EPSFBuilder(oversampling=2, maxiters=3)
            epsf, fitted_stars = epsf_builder(stars_tbl)
            
            if epsf is None:
                return 0.0, 0.0
                
            # Calculate PSF quality (symmetry and smoothness)
            psf_data = epsf.data
            if psf_data is None or not np.any(np.isfinite(psf_data)):
                return 0.0, 0.0
                
            center = psf_data.shape[0] // 2
            
            # Symmetry check
            flipped = np.flip(psf_data)
            peak = np.max(psf_data)
            if peak == 0:
                return 0.0, 0.0
                
            symmetry_score = 1 - np.mean(np.abs(psf_data - flipped)) / peak
            
            # Estimate magnitude using PSF photometry
            magnitude = -2.5 * np.log10(peak)  # Instrumental magnitude
            
            return max(0.0, min(1.0, symmetry_score)), magnitude
            
        except Exception as e:
            logger.error(f"Error analyzing PSF: {str(e)}")
            return 0.0, 0.0

    def calculate_proper_motion(self,
                              coord1: SkyCoord,
                              coord2: SkyCoord,
                              time_diff: float) -> Tuple[float, float]:
        """
        Calculate proper motion and position angle.
        
        Args:
            coord1: First position
            coord2: Second position
            time_diff: Time difference in hours
            
        Returns:
            Tuple of (proper motion in arcsec/hour, position angle in degrees)
        """
        try:
            if coord1 is None or coord2 is None or time_diff == 0:
                return 0.0, 0.0
                
            sep = coord1.separation(coord2)
            pa = coord1.position_angle(coord2)
            
            proper_motion = sep.arcsec / time_diff
            
            return proper_motion, pa.deg
            
        except Exception as e:
            logger.error(f"Error calculating proper motion: {str(e)}")
            return 0.0, 0.0

    def analyze_candidate(self,
                        candidate: Candidate,
                        fits_path1: str,
                        fits_path2: str,
                        observation_time: str) -> Optional[DetailedCandidate]:
        """
        Perform detailed analysis of a candidate.
        
        Args:
            candidate: Candidate object from fast detection
            fits_path1: Path to first FITS file
            fits_path2: Path to second FITS file
            observation_time: Time of first observation
            
        Returns:
            DetailedCandidate object if analysis passes, None otherwise
        """
        try:
            # Get WCS information
            wcs = self.get_wcs(fits_path1)
            if wcs is None:
                return None
            
            # Convert pixel positions to world coordinates
            coord1 = self.pixel_to_world(candidate.x, candidate.y, wcs)
            if coord1 is None:
                return None
                
            coord2 = self.pixel_to_world(
                candidate.x + candidate.movement_px * np.cos(np.radians(candidate.direction)),
                candidate.y + candidate.movement_px * np.sin(np.radians(candidate.direction)),
                wcs
            )
            if coord2 is None:
                return None
            
            # Check for known objects
            is_known = self.check_known_objects(coord1, observation_time)
            
            # Calculate proper motion
            proper_motion, position_angle = self.calculate_proper_motion(
                coord1, coord2, candidate.time_diff
            )
            
            # Analyze PSF
            with fits.open(fits_path1) as hdul:
                if len(hdul) == 0 or hdul[0].data is None:
                    return None
                # Get data and ensure it's 2D
                image_data = hdul[0].data
                
                # If data is 3D or higher, take the first 2D slice
                while image_data.ndim > 2:
                    image_data = image_data[0]
                
                # Scale data similar to FastDetector
                mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
                image_data = (image_data - median)
                if std > 0:
                    image_data = image_data / std
                image_data = np.clip(image_data, 0, 50)
                
            psf_quality, magnitude = self.analyze_psf(
                image_data,
                candidate.x,
                candidate.y
            )
            
            # Calculate confidence score
            confidence_factors = [
                psf_quality,
                min(1.0, proper_motion / 10.0),  # Normalize proper motion
                candidate.snr / 20.0  # Normalize SNR
            ]
            confidence_score = np.mean([f for f in confidence_factors if f > 0])
            
            if confidence_score < self.min_confidence:
                return None
                
            return DetailedCandidate(
                initial_candidate=candidate,
                ra=coord1.ra.deg,
                dec=coord1.dec.deg,
                proper_motion=proper_motion,
                position_angle=position_angle,
                psf_quality=psf_quality,
                known_object=is_known,
                magnitude=magnitude,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing candidate: {str(e)}")
            return None

    def filter_candidates(self,
                        candidates: List[DetailedCandidate]) -> List[DetailedCandidate]:
        """
        Filter candidates based on various criteria.
        
        Args:
            candidates: List of detailed candidates
            
        Returns:
            Filtered list of candidates
        """
        filtered = []
        
        for candidate in candidates:
            # Skip known objects
            if candidate.known_object:
                continue
                
            # Skip low confidence candidates
            if candidate.confidence_score < self.min_confidence:
                continue
                
            # Skip candidates with poor PSF quality
            if candidate.psf_quality < 0.7:
                continue
                
            filtered.append(candidate)
            
        return filtered