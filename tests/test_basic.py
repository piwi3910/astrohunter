"""
Basic tests for the asteroid hunter package.
"""
import os
import pytest
from astrohunter import (
    Downloader,
    FastDetector,
    DetailedAnalyzer,
    AsteroidHunter
)

def test_downloader_initialization():
    """Test that downloader can be initialized."""
    data_dir = "test_data"
    downloader = Downloader(data_dir)
    assert os.path.exists(data_dir)
    os.rmdir(data_dir)

def test_fast_detector_initialization():
    """Test that fast detector can be initialized with custom parameters."""
    detector = FastDetector(
        snr_threshold=4.0,
        min_movement=0.3,
        max_movement=40.0
    )
    assert detector.snr_threshold == 4.0
    assert detector.min_movement == 0.3
    assert detector.max_movement == 40.0

def test_detailed_analyzer_initialization():
    """Test that detailed analyzer can be initialized with custom parameters."""
    analyzer = DetailedAnalyzer(
        psf_size=20,
        match_radius=25.0,
        min_confidence=0.8
    )
    assert analyzer.psf_size == 20
    assert analyzer.match_radius == 25.0
    assert analyzer.min_confidence == 0.8

def test_asteroid_hunter_initialization():
    """Test that main asteroid hunter can be initialized."""
    data_dir = "test_data"
    output_dir = "test_results"
    
    hunter = AsteroidHunter(
        data_dir=data_dir,
        output_dir=output_dir
    )
    
    assert os.path.exists(data_dir)
    assert os.path.exists(output_dir)
    
    # Cleanup
    os.rmdir(data_dir)
    os.rmdir(output_dir)

if __name__ == "__main__":
    pytest.main([__file__])