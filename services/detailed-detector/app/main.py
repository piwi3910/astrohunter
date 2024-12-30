import asyncio
import json
import logging
import os
from pathlib import Path
import aio_pika
import redis
from .analyzer import DetailedAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    decode_responses=True
)

class DetailedDetectorService:
    def __init__(self):
        self.analyzer = DetailedAnalyzer()

    async def connect_rabbitmq(self):
        """Connect to RabbitMQ"""
        return await aio_pika.connect_robust(
            f"amqp://guest:guest@{os.getenv('RABBITMQ_HOST', 'localhost')}/"
        )

    async def process_detection_request(self, message: aio_pika.IncomingMessage):
        """Process incoming detection request"""
        async with message.process():
            try:
                body = json.loads(message.body.decode())
                job_id = body['job_id']
                files = body['files']
                initial_detections = body['initial_detections']
                coords = body['coordinates']

                logger.info(f"Processing detailed detection for job {job_id}")

                # Update Redis status
                redis_client.set(f"job:{job_id}", json.dumps({
                    "status": "detailed_detection",
                    "coordinates": coords
                }))

                confirmed_detections = []
                for detection in initial_detections:
                    # Find corresponding file
                    file_id = detection.get('file_id', 0)
                    if file_id >= len(files):
                        continue

                    # Perform detailed analysis
                    detailed_result = self.analyzer.detailed_analysis(
                        files[file_id],
                        detection
                    )

                    if detailed_result:
                        # Add file information
                        detailed_result['file_id'] = file_id
                        confirmed_detections.append(detailed_result)

                # If we have multiple detections, analyze their trajectory
                if len(confirmed_detections) > 1:
                    trajectory_data = self.analyzer.analyze_trajectory(confirmed_detections)
                    for detection in confirmed_detections:
                        detection['metadata']['trajectory'] = trajectory_data

                # Send results to data service
                connection = await self.connect_rabbitmq()
                async with connection:
                    channel = await connection.channel()
                    queue = await channel.declare_queue("detection_results_queue")
                    
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps({
                                "job_id": job_id,
                                "detections": confirmed_detections,
                                "detection_type": "detailed"
                            }).encode()
                        ),
                        routing_key="detection_results_queue"
                    )

                logger.info(
                    f"Detailed detection completed for job {job_id} "
                    f"with {len(confirmed_detections)} confirmed detections"
                )

            except Exception as e:
                logger.error(f"Error processing detailed detection: {str(e)}")
                # Update Redis with error status
                redis_client.set(f"job:{job_id}", json.dumps({
                    "status": "error",
                    "coordinates": coords,
                    "results": {"error": str(e)}
                }))

    async def run(self):
        """Run the detailed detector service"""
        # Connect to RabbitMQ
        connection = await self.connect_rabbitmq()
        async with connection:
            # Create channel
            channel = await connection.channel()
            
            # Declare queue
            queue = await channel.declare_queue("detailed_detector_queue")
            
            # Start consuming messages
            logger.info("Detailed detector service started, waiting for messages...")
            await queue.consume(self.process_detection_request)
            
            try:
                await asyncio.Future()  # run forever
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    service = DetailedDetectorService()
    asyncio.run(service.run())