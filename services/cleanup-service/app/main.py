import asyncio
import json
import logging
import os
from pathlib import Path
import aio_pika
import redis
import schedule
import time
from datetime import datetime, timedelta
import aiofiles
import shutil
from typing import List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/cleanup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    decode_responses=True
)

class CleanupService:
    def __init__(self):
        self.data_dir = Path("/app/data")
        self.retention_days = int(os.getenv("DATA_RETENTION_DAYS", "7"))
        self.min_disk_space_gb = float(os.getenv("MIN_DISK_SPACE_GB", "10"))
        self.cleanup_schedule_hours = int(os.getenv("CLEANUP_SCHEDULE_HOURS", "24"))

    async def connect_rabbitmq(self):
        """Connect to RabbitMQ"""
        return await aio_pika.connect_robust(
            f"amqp://guest:guest@{os.getenv('RABBITMQ_HOST', 'localhost')}/"
        )

    def get_disk_space(self, path: str) -> tuple:
        """Get available disk space in GB"""
        stats = shutil.disk_usage(path)
        return (
            stats.total / (1024**3),
            stats.used / (1024**3),
            stats.free / (1024**3)
        )

    async def get_active_job_files(self) -> Set[str]:
        """Get list of files from active jobs"""
        active_files = set()
        for key in redis_client.scan_iter("job:*"):
            job_data = redis_client.get(key)
            if job_data:
                try:
                    data = json.loads(job_data)
                    if data.get('status') in ['processing', 'downloading', 'fast_detection', 'detailed_detection']:
                        files = data.get('files', [])
                        active_files.update(files)
                except json.JSONDecodeError:
                    continue
        return active_files

    async def cleanup_old_data(self):
        """Clean up old data files"""
        try:
            logger.info("Starting scheduled cleanup")
            
            # Check disk space
            total, used, free = self.get_disk_space("/app/data")
            logger.info(f"Disk space - Total: {total:.2f}GB, Used: {used:.2f}GB, Free: {free:.2f}GB")

            # Get active job files
            active_files = await self.get_active_job_files()
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            # Walk through data directory
            files_cleaned = 0
            bytes_cleaned = 0
            
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    file_path = Path(root) / file
                    
                    # Skip if file is being used by active job
                    if str(file_path) in active_files:
                        continue
                    
                    try:
                        # Check file age
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mtime < cutoff_date:
                            size = file_path.stat().st_size
                            file_path.unlink()
                            files_cleaned += 1
                            bytes_cleaned += size
                            logger.info(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")

            # Clean up empty directories
            for root, dirs, _ in os.walk(self.data_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    try:
                        dir_path.rmdir()  # Will only succeed if directory is empty
                        logger.info(f"Removed empty directory: {dir_path}")
                    except OSError:
                        pass  # Directory not empty

            # Log cleanup results
            logger.info(
                f"Cleanup completed - Files: {files_cleaned}, "
                f"Space freed: {bytes_cleaned / (1024**3):.2f}GB"
            )

            # Check if emergency cleanup needed
            if free < self.min_disk_space_gb:
                logger.warning("Low disk space detected, initiating emergency cleanup")
                await self.emergency_cleanup()

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def emergency_cleanup(self):
        """Perform emergency cleanup when disk space is critically low"""
        try:
            # Get active job files
            active_files = await self.get_active_job_files()
            
            # Sort all files by size and age
            file_info = []
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    file_path = Path(root) / file
                    if str(file_path) not in active_files:
                        try:
                            stats = file_path.stat()
                            file_info.append({
                                'path': file_path,
                                'size': stats.st_size,
                                'mtime': stats.st_mtime
                            })
                        except Exception:
                            continue

            # Sort by age (oldest first)
            file_info.sort(key=lambda x: x['mtime'])
            
            # Delete files until we reach target free space
            for file_data in file_info:
                try:
                    file_data['path'].unlink()
                    logger.info(f"Emergency cleanup: removed {file_data['path']}")
                    
                    # Check if we've freed enough space
                    _, _, free = self.get_disk_space("/app/data")
                    if free >= self.min_disk_space_gb:
                        break
                except Exception as e:
                    logger.error(f"Error removing file {file_data['path']}: {str(e)}")

        except Exception as e:
            logger.error(f"Error during emergency cleanup: {str(e)}")

    async def run(self):
        """Run the cleanup service"""
        logger.info("Starting cleanup service")

        # Schedule regular cleanup
        schedule.every(self.cleanup_schedule_hours).hours.do(
            lambda: asyncio.create_task(self.cleanup_old_data())
        )

        # Initial cleanup
        await self.cleanup_old_data()

        # Run scheduler
        while True:
            schedule.run_pending()
            await asyncio.sleep(60)  # Check schedule every minute

if __name__ == "__main__":
    service = CleanupService()
    asyncio.run(service.run())