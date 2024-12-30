import asyncio
import json
import logging
import os
from pathlib import Path
import aio_pika
import redis
from itertools import pairwise
from .detector import FastDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    decode_responses=True
)

class FastDetectorService:
    def __init__(self):
        self.detector = FastDetector()

    async def connect_rabbitmq(self):
        """Connect to RabbitMQ"""
        return await aio_pika.connect_robust(
            f"amqp://guest:guest@{os.getenv('RABBITMQ_HOST', 'localhost')}/"
        )

    def group_files_by_time(self, files: list) -> list:
        """Group files by observation time"""
        # Sort files by timestamp in filename
        sorted_files = sorted(files)
        
        # Group consecutive files into pairs
        return list(pairwise(sorted_files))

    async def process_detection_request(self, message: aio_pika.IncomingMessage):
        """Process incoming detection request"""
        async with message.process():
            try:
                body = json.loads(message.body.decode())
                job_id = body['job_id']
                files = body['files']
                coords = body['coordinates']

                logger.info(f"Processing detection request for job {job_id}")

                # Update Redis status
                redis_client.set(f"job:{job_id}", json.dumps({
                    "status": "fast_detection",
                    "coordinates": coords
                }))

                # Group files into pairs for analysis
                file_pairs = self.group_files_by_time(files)
                
                all_detections = []
                for file1, file2 in file_pairs:
                    # Analyze image pair
                    detections = self.detector.analyze_image_pair(file1, file2)
                    
                    if detections:
                        # Add file information to detections
                        for detection in detections:
                            detection['file_id'] = files.index(file2)  # Use second file as reference
                        all_detections.extend(detections)

                # Send results to data service
                connection = await self.connect_rabbitmq()
                async with connection:
                    channel = await connection.channel()
                    
                    # Send to data service
                    data_queue = await channel.declare_queue("detection_results_queue")
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps({
                                "job_id": job_id,
                                "detections": all_detections,
                                "detection_type": "fast"
                            }).encode()
                        ),
                        routing_key="detection_results_queue"
                    )

                    # Send to detailed detector if any detections found
                    if all_detections:
                        detailed_queue = await channel.declare_queue("detailed_detector_queue")
                        await channel.default_exchange.publish(
                            aio_pika.Message(
                                body=json.dumps({
                                    "job_id": job_id,
                                    "files": files,
                                    "coordinates": coords,
                                    "initial_detections": all_detections
                                }).encode()
                            ),
                            routing_key="detailed_detector_queue"
                        )

                logger.info(f"Fast detection completed for job {job_id} with {len(all_detections)} detections")

            except Exception as e:
                logger.error(f"Error processing detection request: {str(e)}")
                # Update Redis with error status
                redis_client.set(f"job:{job_id}", json.dumps({
                    "status": "error",
                    "coordinates": coords,
                    "results": {"error": str(e)}
                }))

    async def run(self):
        """Run the fast detector service"""
        # Connect to RabbitMQ
        connection = await self.connect_rabbitmq()
        async with connection:
            # Create channel
            channel = await connection.channel()
            
            # Declare queue
            queue = await channel.declare_queue("fast_detector_queue")
            
            # Start consuming messages
            logger.info("Fast detector service started, waiting for messages...")
            await queue.consume(self.process_detection_request)
            
            try:
                await asyncio.Future()  # run forever
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    service = FastDetectorService()
    asyncio.run(service.run())