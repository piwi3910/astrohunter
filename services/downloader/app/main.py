import asyncio
import json
import os
import logging
from pathlib import Path
import aio_pika
import redis
from astroquery.mast import Observations
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    decode_responses=True
)

class DownloaderService:
    def __init__(self):
        self.data_dir = Path("/app/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    async def connect_rabbitmq(self):
        """Connect to RabbitMQ"""
        return await aio_pika.connect_robust(
            f"amqp://guest:guest@{os.getenv('RABBITMQ_HOST', 'localhost')}/"
        )

    async def get_field_observations(self, ra: float, dec: float, days_back: int = 7) -> pd.DataFrame:
        """Get observations for a single field"""
        try:
            obs_table = Observations.query_criteria(
                dataproduct_type='image',
                s_ra=[ra - 1.0, ra + 1.0],  # 2.0 degree range
                s_dec=[dec - 1.0, dec + 1.0],  # 2.0 degree range
                obs_collection=['HST']
            )

            if obs_table is None or len(obs_table) == 0:
                return pd.DataFrame()

            df = pd.DataFrame({
                'obsid': obs_table['obsid'].astype(str),
                'obsmjd': obs_table['t_min'].astype(float)
            })
            
            df.sort_values('obsmjd', inplace=True)
            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error getting field observations: {str(e)}")
            return pd.DataFrame()

    async def download_data(self, obs_ids: list, job_id: str) -> list:
        """Download data for the given observation IDs"""
        try:
            products = Observations.get_product_list(obs_ids)
            
            # Filter by size and type
            size_mask = products['size'] < (1500 * 1024 * 1024)  # 1.5GB limit
            products = products[size_mask]
            
            images = Observations.filter_products(
                products,
                productType='SCIENCE',
                extension='fits'
            )

            if len(images) == 0:
                logger.warning("No suitable FITS images found")
                return []

            # Create job-specific directory
            job_dir = self.data_dir / job_id
            job_dir.mkdir(exist_ok=True)

            manifest = Observations.download_products(
                images,
                download_dir=str(job_dir),
                cache=False
            )

            return manifest['Local Path'].tolist()

        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            return []

    async def process_download_request(self, message: aio_pika.IncomingMessage):
        """Process download request from queue"""
        async with message.process():
            try:
                body = json.loads(message.body.decode())
                job_id = body['job_id']
                coords = body['coordinates']
                
                # Update job status
                redis_client.set(f"job:{job_id}", json.dumps({
                    "status": "downloading",
                    "coordinates": coords
                }))

                # Get observations
                df = await self.get_field_observations(coords['ra'], coords['dec'])
                
                if df.empty:
                    logger.warning(f"No observations found for job {job_id}")
                    redis_client.set(f"job:{job_id}", json.dumps({
                        "status": "completed",
                        "coordinates": coords,
                        "results": {"error": "No observations found"}
                    }))
                    return

                # Download data
                downloaded_files = await self.download_data(df['obsid'].tolist(), job_id)

                if not downloaded_files:
                    logger.warning(f"No files downloaded for job {job_id}")
                    redis_client.set(f"job:{job_id}", json.dumps({
                        "status": "completed",
                        "coordinates": coords,
                        "results": {"error": "No files could be downloaded"}
                    }))
                    return

                # Send message to data service
                connection = await self.connect_rabbitmq()
                async with connection:
                    channel = await connection.channel()
                    data_queue = await channel.declare_queue("data_service_queue")
                    
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps({
                                "job_id": job_id,
                                "files": downloaded_files,
                                "coordinates": coords
                            }).encode()
                        ),
                        routing_key="data_service_queue"
                    )

                logger.info(f"Download completed for job {job_id}")

            except Exception as e:
                logger.error(f"Error processing download request: {str(e)}")
                redis_client.set(f"job:{job_id}", json.dumps({
                    "status": "error",
                    "coordinates": coords,
                    "results": {"error": str(e)}
                }))

    async def run(self):
        """Run the downloader service"""
        # Connect to RabbitMQ
        connection = await self.connect_rabbitmq()
        async with connection:
            # Create channel
            channel = await connection.channel()
            
            # Declare queue
            queue = await channel.declare_queue("download_queue")
            
            # Start consuming messages
            logger.info("Downloader service started, waiting for messages...")
            await queue.consume(self.process_download_request)
            
            try:
                await asyncio.Future()  # run forever
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    service = DownloaderService()
    asyncio.run(service.run())