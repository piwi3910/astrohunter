"""
Main script for running the asteroid detection pipeline.
"""
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import List
from dataclasses import asdict
import sys
import gc

from astrohunter.downloader import Downloader
from astrohunter.fast_detector import FastDetector
from astrohunter.detailed_analyzer import DetailedAnalyzer, DetailedCandidate

def setup_logging(debug: bool = False):
    """Set up logging configuration."""
    # Configure root logger first
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Set high threshold for root

    # Create our own logger for astrohunter
    logger = logging.getLogger('astrohunter')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(message)s')

    # Create and configure stream handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Remove any existing handlers and add our new one
    logger.handlers = []
    logger.addHandler(handler)

    # Ensure propagation is off to avoid duplicate messages
    logger.propagate = False

# Set up module logger
logger = logging.getLogger('astrohunter')

class AsteroidHunter:
    def __init__(self,
                 data_dir: str = "data",
                 output_dir: str = "results"):
        """
        Initialize the asteroid hunting pipeline.

        Args:
            data_dir: Directory for storing downloaded data
            output_dir: Directory for storing results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing asteroid hunter")

        # Initialize components
        self.downloader = Downloader(str(self.data_dir))
        self.fast_detector = FastDetector()
        self.detailed_analyzer = DetailedAnalyzer()

    def save_results(self,
                     candidates: List[DetailedCandidate],
                     output_file: str = None):
        """
        Save detection results to JSON file.

        Args:
            candidates: List of detailed candidates
            output_file: Optional output filename
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"candidates_{timestamp}.json"

        output_path = self.output_dir / output_file

        # Convert candidates to dictionary
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_candidates": len(candidates),
            "candidates": [asdict(c) for c in candidates]
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Found {len(candidates)} candidates")

    def process_field_pair(self,
                           obs_id1: str,
                           obs_id2: str) -> List[DetailedCandidate]:
        """
        Process a pair of observations for asteroid detection.

        Args:
            obs_id1: First observation ID
            obs_id2: Second observation ID

        Returns:
            List of detailed candidates
        """
        try:
            # Get metadata first to get coordinates
            meta1 = self.downloader.get_field_metadata(obs_id1)
            meta2 = self.downloader.get_field_metadata(obs_id2)

            # Download data for the observation IDs
            files = self.downloader.download_data([obs_id1, obs_id2], max_size_mb=1500)  # Allow up to 1.5GB files
            if not files or len(files) < 2:
                logger.error(f"Failed to download data for observation IDs {obs_id1} and {obs_id2}")
                return []

            # Download observations as cutouts
            file1 = self.downloader.get_cutout(files[0], meta1.get('ra'), meta1.get('dec'))
            file2 = self.downloader.get_cutout(files[1], meta2.get('ra'), meta2.get('dec'))

            if file1 is None or file2 is None:
                return []

            # Calculate time difference
            time1 = datetime.now()  # Placeholder, replace with actual time from metadata
            time2 = datetime.now()  # Placeholder, replace with actual time from metadata
            if 'time' in meta1 and 'time' in meta2:
                time1 = datetime.fromisoformat(meta1['time'])
                time2 = datetime.fromisoformat(meta2['time'])
            time_diff_hours = (time2 - time1).total_seconds() / 3600

            # Fast detection
            candidates = self.fast_detector.find_candidates(file1, file2, time_diff_hours)
            if not candidates:
                return []

            logger.debug(f"Found {len(candidates)} initial candidates")

            # Detailed analysis
            detailed_candidates = []
            for candidate in candidates:
                result = self.detailed_analyzer.analyze_candidate(
                    candidate,
                    file1,
                    file2,
                    meta1.get('time', datetime.now().isoformat())
                )
                if result is not None:
                    detailed_candidates.append(result)

            # Filter candidates
            filtered_candidates = self.detailed_analyzer.filter_candidates(detailed_candidates)
            if filtered_candidates:
                logger.info(f"Found {len(filtered_candidates)} promising candidates")

            return filtered_candidates

        except Exception as e:
            logger.error(f"Error processing pair: {str(e)}")
            return []
        finally:
            gc.collect()

    def run(self, days_back: int = 1):
        """
        Run the complete asteroid detection pipeline.

        Args:
            days_back: Number of days back to search for observations
        """
        logger.info(f"Starting asteroid detection pipeline (looking back {days_back} days)")

        # Define test fields (RA, Dec)
        test_fields = [
            (83.8, -5.4),     # Orion Nebula
            (10.7, 41.3),     # Andromeda Galaxy
            (266.4, -29.0)    # Galactic Center
        ]

        # Process pairs using generator
        all_candidates = []
        processed = 0

        # Get observation pairs using generator
        for obs_id1, obs_id2 in self.downloader.get_field_pairs(
            ra_list=test_fields,
            days_back=days_back,
            time_window=2.0  # Allow up to 2 hours between observations
        ):
            processed += 1
            if processed % 10 == 0:  # Show progress every 10 pairs
                logger.info(f"Progress: {processed} pairs processed")
                gc.collect()  # Regular cleanup

            candidates = self.process_field_pair(obs_id1, obs_id2)
            all_candidates.extend(candidates)

        # Save results
        self.save_results(all_candidates)
        logger.info(f"Pipeline complete - found {len(all_candidates)} total candidates")

def main():
    """Command line interface for the asteroid hunter."""
    parser = argparse.ArgumentParser(description="Hunt for unknown asteroids using public data")

    parser.add_argument(
        "--days",
        type=int,
        default=7,  # Look back a week by default for HST data
        help="Number of days back to search"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for storing downloaded data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for storing results"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(debug=args.debug)

    # Run the pipeline
    hunter = AsteroidHunter(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    hunter.run(days_back=args.days)

if __name__ == "__main__":
    main()