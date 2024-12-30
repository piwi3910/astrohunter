import asyncio
import json
import logging
import os
from pathlib import Path
import aio_pika
import redis
from astropy.io import fits
from .database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    decode_responses=True
)

class DataService:
    def __init__(self):
        self.db = DatabaseManager()
        self.data_dir = Path("/app/data")

    async def connect_rabbitmq(self):
        """Connect to RabbitMQ"""
        return await aio_pika.connect_robust(
            f"amqp://guest:guest@{os.getenv('RABBITMQ_HOST', 'localhost')}/"
        )

    async def extract_fits_metadata(self, file_path: str) -> dict:
        """Extract metadata from FITS file"""
        try:
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                return {
                    'TELESCOP': header.get('TELESCOP'),
                    'INSTRUME': header.get('INSTRUME'),
                    'DATE-OBS': header.get('DATE-OBS'),
                    'EXPTIME': header.get('EXPTIME'),
                    'FILTER': header.get('FILTER'),
                    'IMAGETYP': header.get('IMAGETYP'),
                    'OBJECT': header.get('OBJECT'),
                    'RA': header.get('RA'),
                    'DEC': header.get('DEC'),
                }
        except Exception as e:
            logger.error(f"Error extracting FITS metadata: {str(e)}")
            return {}

    async def process_data_files(self, message: aio_pika.IncomingMessage):
        """Process incoming data files message"""
        async with message.process():
            try:
                body = json.loads(message.body.decode())
                job_id = body['job_id']
                files = body['files']
                coords = body['coordinates']

                # Update job status
                await self.db.update_job_status(job_id, "processing")
                redis_client.set(f"job:{job_id}", json.dumps({
                    "status": "processing",
                    "coordinates": coords
                }))

                # Process each file
                for file_path in files:
                    # Extract metadata
                    metadata = await self.extract_fits_metadata(file_path)
                    
                    # Add file to database
                    await self.db.add_files(job_id, [file_path], metadata)

                # Send message to fast detector service
                connection = await self.connect_rabbitmq()
                async with connection:
                    channel = await connection.channel()
                    fast_detector_queue = await channel.declare_queue("fast_detector_queue")
                    
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps({
                                "job_id": job_id,
                                "files": files,
                                "coordinates": coords
                            }).encode()
                        ),
                        routing_key="fast_detector_queue"
                    )

                logger.info(f"Files processed for job {job_id}")

            except Exception as e:
                logger.error(f"Error processing data files: {str(e)}")
                await self.db.update_job_status(job_id, "error", {"error": str(e)})
                redis_client.set(f"job:{job_id}", json.dumps({
                    "status": "error",
                    "coordinates": coords,
                    "results": {"error": str(e)}
                }))

    async def handle_detection_results(self, message: aio_pika.IncomingMessage):
        """Handle detection results from detector services"""
        async with message.process():
            try:
                body = json.loads(message.body.decode())
                job_id = body['job_id']
                detections = body['detections']
                detection_type = body['detection_type']

                # Add detections to database
                for detection in detections:
                    await self.db.add_detection(
                        job_id=job_id,
                        file_id=detection['file_id'],
                        detection_data={
                            'ra': detection['ra'],
                            'dec': detection['dec'],
                            'magnitude': detection.get('magnitude'),
                            'type': detection_type,
                            'confidence': detection.get('confidence'),
                            'metadata': detection.get('metadata')
                        }
                    )

                # If this is from detailed detector, update job status to completed
                if detection_type == 'detailed':
                    all_detections = await self.db.get_job_detections(job_id)
                    await self.db.update_job_status(job_id, "completed", {
                        "total_detections": len(all_detections)
                    })
                    
                    # Update Redis with final results
                    redis_client.set(f"job:{job_id}", json.dumps({
                        "status": "completed",
                        "results": {
                            "total_detections": len(all_detections),
                            "detections": [
                                {
                                    "ra": d.ra,
                                    "dec": d.dec,
                                    "magnitude": d.magnitude,
                                    "confidence": d.confidence
                                }
                                for d in all_detections
                            ]
                        }
                    }))

                logger.info(f"Processed {detection_type} detections for job {job_id}")

            except Exception as e:
                logger.error(f"Error handling detection results: {str(e)}")
                await self.db.update_job_status(job_id, "error", {"error": str(e)})

    async def run(self):
        """Run the data service"""
        # Initialize database
        await self.db.init_db()

        # Connect to RabbitMQ
        connection = await self.connect_rabbitmq()
        async with connection:
            # Create channel
            channel = await connection.channel()
            
            # Declare queues
            data_queue = await channel.declare_queue("data_service_queue")
            detection_results_queue = await channel.declare_queue("detection_results_queue")
            
            # Start consuming messages
            await data_queue.consume(self.process_data_files)
            await detection_results_queue.consume(self.handle_detection_results)
            
            logger.info("Data service started, waiting for messages...")
            
            try:
                await asyncio.Future()  # run forever
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    service = DataService()
    asyncio.run(service.run())