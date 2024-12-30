from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aio_pika
import redis
import json
import os
import uuid
from typing import List, Optional

app = FastAPI(title="AstroHunter API Gateway")

# Models
class Coordinates(BaseModel):
    ra: float
    dec: float

class JobStatus(BaseModel):
    job_id: str
    status: str
    results: Optional[dict] = None

# Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    decode_responses=True
)

# RabbitMQ connection
async def get_rabbitmq_connection():
    return await aio_pika.connect_robust(
        f"amqp://guest:guest@{os.getenv('RABBITMQ_HOST', 'localhost')}/"
    )

@app.post("/search", response_model=dict)
async def submit_search(coords: Coordinates):
    """Submit coordinates for asteroid search"""
    job_id = str(uuid.uuid4())
    
    # Store initial job status in Redis
    job_data = {
        "status": "submitted",
        "coordinates": coords.dict(),
        "results": None
    }
    redis_client.set(f"job:{job_id}", json.dumps(job_data))
    
    # Send message to downloader service
    connection = await get_rabbitmq_connection()
    async with connection:
        channel = await connection.channel()
        
        # Declare the download queue
        download_queue = await channel.declare_queue("download_queue")
        
        # Create the message
        message = {
            "job_id": job_id,
            "coordinates": coords.dict()
        }
        
        # Send the message
        await channel.default_exchange.publish(
            aio_pika.Message(body=json.dumps(message).encode()),
            routing_key="download_queue"
        )
    
    return {"job_id": job_id, "status": "submitted"}

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a submitted job"""
    job_data = redis_client.get(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = json.loads(job_data)
    return JobStatus(
        job_id=job_id,
        status=job_info["status"],
        results=job_info.get("results")
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)