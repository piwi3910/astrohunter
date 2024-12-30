"""
Test minimal download functionality.
"""
import os
import shutil
from pathlib import Path
from astroquery.mast import Observations
import psutil

def get_memory_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_minimal_download():
    """Test downloading a single file with memory tracking."""
    # Create test directory
    test_dir = Path("test_downloads")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Track initial memory
        initial_mem = get_memory_mb()
        print(f"\nInitial memory: {initial_mem:.1f}MB")
        
        # Query with specific criteria
        print("\nQuerying observations...")
        mem_before = get_memory_mb()
        
        # Use specific HST observation criteria
        obs_table = Observations.query_criteria(
            obs_collection='HST',
            proposal_id='15676',  # Known HST proposal for Orion
            dataproduct_type='image'
        )
        print(f"Memory after query: {get_memory_mb():.1f}MB (delta: {get_memory_mb() - mem_before:.1f}MB)")
        
        if obs_table is None or len(obs_table) == 0:
            print("No observations found")
            return
            
        print(f"Found {len(obs_table)} observations")
            
        # Get first observation ID
        obs_id = obs_table[0]['obsid']
        print(f"\nUsing observation ID: {obs_id}")
        
        # Get products
        print("\nGetting product list...")
        mem_before = get_memory_mb()
        products = Observations.get_product_list(obs_id)
        print(f"Memory after product list: {get_memory_mb():.1f}MB (delta: {get_memory_mb() - mem_before:.1f}MB)")
        
        if products is None or len(products) == 0:
            print("No products found")
            return
            
        print(f"Found {len(products)} products")
        
        # Filter for smallest FITS file
        print("\nFiltering for FITS files...")
        mem_before = get_memory_mb()
        fits_products = Observations.filter_products(
            products,
            productType='SCIENCE',
            extension='fits'
        )
        print(f"Memory after filter: {get_memory_mb():.1f}MB (delta: {get_memory_mb() - mem_before:.1f}MB)")
        
        if len(fits_products) == 0:
            print("No FITS files found")
            return
            
        print(f"Found {len(fits_products)} FITS files")
            
        # Sort by size and get smallest
        fits_products.sort('size')
        smallest_product = fits_products[0:1]
        size_mb = float(smallest_product['size'][0]) / 1024 / 1024
        print(f"Selected smallest FITS file: {size_mb:.1f}MB")
        
        # Download file
        print("\nDownloading file...")
        mem_before = get_memory_mb()
        manifest = Observations.download_products(
            smallest_product,
            download_dir=str(test_dir),
            cache=True,
            curl_flag=False
        )
        print(f"Memory after download: {get_memory_mb():.1f}MB (delta: {get_memory_mb() - mem_before:.1f}MB)")
        
        if manifest is None or len(manifest) == 0:
            print("Download failed")
            return
            
        # Verify download
        downloaded_file = manifest['Local Path'][0]
        actual_size = os.path.getsize(downloaded_file) / 1024 / 1024
        print(f"\nDownloaded: {downloaded_file} ({actual_size:.1f}MB)")
        
        # Final memory check
        final_mem = get_memory_mb()
        print(f"\nFinal memory: {final_mem:.1f}MB")
        print(f"Total memory increase: {final_mem - initial_mem:.1f}MB")
        
    finally:
        # Cleanup
        print("\nCleaning up...")
        if test_dir.exists():
            try:
                shutil.rmtree(test_dir)
            except Exception as e:
                print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    test_minimal_download()