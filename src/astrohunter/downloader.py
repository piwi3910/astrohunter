"""
Module for downloading astronomical data using astroquery.
"""
import logging
from typing import List, Optional, Tuple, Generator
from pathlib import Path
import gc

import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from astroquery.mast import Observations
from astroquery.exceptions import TimeoutError, InvalidQueryError

# Configure module logger
logger = logging.getLogger('astrohunter')

class Downloader:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the downloader with a directory for storing data.

        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure Observations
        Observations.TIMEOUT = 60  # 1 minute timeout

    def _get_field_observations(self, ra: float, dec: float, days_back: int = 1) -> pd.DataFrame:
        """
        Get observations for a single field with memory-efficient filtering.
        Uses a 1.0 degree search radius to find observations containing the target field,
        which helps ensure we get full-frame images that include the area of interest.
        
        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            days_back: Number of days back to search
            
        Returns:
            DataFrame with filtered observations
        """
        try:
            # Query for HST observations in the field
            obs_table = Observations.query_criteria(
                obs_collection=['HST'],
                dataproduct_type='image',
                s_ra=[ra - 0.5, ra + 0.5],  # 1.0 degree range
                s_dec=[dec - 0.5, dec + 0.5]  # 1.0 degree range
            )

            if obs_table is None or len(obs_table) == 0:
                return pd.DataFrame()

            # Convert only needed columns to DataFrame
            df = pd.DataFrame({
                'obsid': obs_table['obsid'].astype(str),
                'obsmjd': obs_table['t_min'].astype(float)
            })
            
            # Sort by time
            df.sort_values('obsmjd', inplace=True)
            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error getting field observations: {str(e)}")
            return pd.DataFrame()
        finally:
            gc.collect()

    def get_field_pairs(self, ra_list: List[Tuple[float, float]], days_back: int = 1,
                       time_window: float = 1.0) -> Generator[Tuple[str, str], None, None]:
        """
        Find pairs of observations of the same field within a time window.
        Memory-efficient generator version.

        Args:
            ra_list: List of tuples containing RA and Dec coordinates
            days_back: Number of days back from now to search for observations
            time_window: Time window in hours to look for pairs

        Yields:
            Tuples containing pairs of observation IDs
        """
        total_pairs = 0
        
        for ra, dec in ra_list:
            try:
                logger.debug(f"Processing field at RA={ra:.4f}, Dec={dec:.4f}")
                
                # Get observations for this field
                df = self._get_field_observations(ra, dec, days_back)
                
                if df.empty:
                    logger.debug(f"No observations found at RA={ra:.4f}, Dec={dec:.4f}")
                    continue

                # Find pairs within time window
                times = df['obsmjd'].values
                obs_ids = df['obsid'].values
                
                for i in range(len(times) - 1):
                    time_diff = (times[i + 1] - times[i]) * 24  # Convert days to hours
                    if time_diff <= time_window:
                        total_pairs += 1
                        yield (obs_ids[i], obs_ids[i + 1])
                        logger.debug(f"Found pair with time diff {time_diff:.2f} hours")

            except Exception as e:
                logger.error(f"Error processing field at RA={ra}, Dec={dec}: {str(e)}")
                continue
            finally:
                gc.collect()

        logger.info(f"Found {total_pairs} total observation pairs")

    def download_data(self, obs_ids: List[str], max_size_mb: int = 1500) -> List[str]:
        """
        Download data for the given observation IDs with size limit.

        Args:
            obs_ids: List of observation IDs
            max_size_mb: Maximum size per file in MB (default 1.5GB). Files are processed
                        one at a time using memory mapping to minimize memory usage.

        Returns:
            List of file paths to downloaded data
        """
        try:
            logger.info(f"Downloading data for observations: {obs_ids}")

            max_retries = 3
            retry_delay = 5  # seconds
            
            # Get product list with retry logic
            for attempt in range(max_retries):
                try:
                    products = Observations.get_product_list(obs_ids)
                    if products is None or len(products) == 0:
                        logger.warning("No products found")
                        return []
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to get product list after {max_retries} attempts: {str(e)}")
                        return []
                    logger.warning(f"Product list attempt {attempt + 1} failed, retrying...")
                    time.sleep(retry_delay)

            # Filter by size and type
            size_mask = products['size'] < (max_size_mb * 1024 * 1024)  # Convert MB to bytes
            products = products[size_mask]
            
            # Filter for science images
            products = products[size_mask]
            images = Observations.filter_products(
                products,
                productType='SCIENCE',
                extension='fits'
            )

            if len(images) == 0:
                logger.warning("No suitable FITS images found")
                return []

            # Sort by size and get smallest files first
            images.sort('size')

            # Download with retry logic
            for attempt in range(max_retries):
                try:
                    manifest = Observations.download_products(
                        images,
                        download_dir=str(self.data_dir),
                        cache=True,
                        curl_flag=False
                    )
                    
                    if manifest is None or len(manifest) == 0:
                        if attempt == max_retries - 1:
                            logger.warning("Download failed after all attempts")
                            return []
                        continue
                    
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to download after {max_retries} attempts: {str(e)}")
                        return []
                    logger.warning(f"Download attempt {attempt + 1} failed, retrying...")
                    time.sleep(retry_delay)

            return manifest['Local Path'].tolist()

        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            return []
        finally:
            gc.collect()

    def get_cutout(self, file_path: str, ra: float, dec: float, size: int = 240) -> Optional[str]:
        """
        Extract a cutout from a FITS image.

        Args:
            file_path: Path to the FITS file
            ra: Right ascension in degrees
            dec: Declination in degrees
            size: Size of cutout in pixels

        Returns:
            Path to the cutout FITS file
        """
        try:
            output_path = self.data_dir / f"cutout_{Path(file_path).name}"

            # Use memory mapping for reading
            with fits.open(file_path, memmap=True, mode='readonly') as hdul:
                # Find HDU with image data
                image_hdu = None
                for hdu in hdul:
                    if isinstance(hdu, fits.ImageHDU) or isinstance(hdu, fits.PrimaryHDU):
                        if hdu.data is not None and hdu.header.get('NAXIS', 0) >= 2:
                            image_hdu = hdu
                            break

                if image_hdu is None:
                    logger.error("No suitable image data found")
                    return None

                # Get WCS
                wcs = WCS(image_hdu.header)
                if not wcs.has_celestial:
                    logger.error("No celestial WCS found")
                    return None

                # Calculate pixel coordinates
                # Add a dummy third coordinate (0) for 3D WCS systems
                coords = np.array([[ra, dec, 0]])
                x, y = wcs.all_world2pix(coords, 0)[0][:2]  # Only take x,y coordinates
                x, y = int(round(x)), int(round(y))
                half_size = size // 2

                # Calculate bounds
                y_min = max(0, y - half_size)
                y_max = min(image_hdu.data.shape[0], y + half_size)
                x_min = max(0, x - half_size)
                x_max = min(image_hdu.data.shape[1], x + half_size)

                # Create cutout header
                header = image_hdu.header.copy()
                header['CRPIX1'] -= x_min
                header['CRPIX2'] -= y_min

                # Write cutout directly to disk
                fits.PrimaryHDU(
                    data=image_hdu.data[y_min:y_max, x_min:x_max],
                    header=header
                ).writeto(output_path, overwrite=True)

            return str(output_path)

        except Exception as e:
            logger.error(f"Error creating cutout: {str(e)}")
            return None
        finally:
            gc.collect()

    def get_field_metadata(self, obs_id: str) -> dict:
        """
        Get metadata for a specific observation.

        Args:
            obs_id: Observation ID

        Returns:
            Dictionary containing metadata
        """
        try:
            # Query with minimal columns and retry logic
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    obs_table = Observations.query_criteria(
                        obsid=obs_id,
                        obs_collection=['HST']
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to query MAST after {max_retries} attempts: {str(e)}")
                        return {}
                    logger.warning(f"MAST query attempt {attempt + 1} failed, retrying...")
                    time.sleep(retry_delay)

            if len(obs_table) == 0:
                logger.warning(f"No metadata found for observation ID {obs_id}")
                return {}

            row = obs_table[0]
            return {
                'ra': float(row['s_ra']),
                'dec': float(row['s_dec']),
                'time': Time(float(row['t_min']), format='mjd').iso,
                'obs_id': obs_id,
            }

        except Exception as e:
            logger.error(f"Error retrieving metadata: {str(e)}")
            return {}
        finally:
            gc.collect()