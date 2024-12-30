# AstroHunter Microservices

A distributed system for detecting asteroids in astronomical images, built with a microservices architecture.

## Architecture

The system consists of the following microservices:

- **API Gateway**: Handles incoming requests and coordinates with other services
- **Downloader Service**: Downloads astronomical data from various sources
- **Data Service**: Manages downloaded files and their metadata
- **Fast Detector**: Performs initial rapid detection of potential asteroids
- **Detailed Detector**: Performs thorough analysis of potential detections
- **Cleanup Service**: Manages data retention and cleanup

### Infrastructure Components:

- **Redis**: Used for caching and real-time status updates
- **RabbitMQ**: Message broker for inter-service communication
- **SQLite**: Database for storing metadata and detection results

## Prerequisites

- Docker
- Docker Compose
- At least 10GB of free disk space

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/astrohunter.git
cd astrohunter
```

2. Build and start the services:
```bash
docker-compose up --build
```

3. The API will be available at `http://localhost:8000`

## API Endpoints

### Submit Search
```http
POST /search
Content-Type: application/json

{
    "ra": 83.8000,
    "dec": -5.4000
}
```

### Check Status
```http
GET /status/{job_id}
```

## Service Communication Flow

1. Client submits coordinates through API Gateway
2. API Gateway creates job and sends to Downloader Service
3. Downloader Service downloads data and notifies Data Service
4. Data Service processes files and notifies Fast Detector
5. Fast Detector performs initial analysis and sends potential detections to Detailed Detector
6. Detailed Detector performs thorough analysis and sends results back to Data Service
7. Client can check status through API Gateway at any time

## Environment Variables

Each service can be configured using environment variables:

```env
REDIS_HOST=redis
RABBITMQ_HOST=rabbitmq
DATA_RETENTION_DAYS=7
MIN_DISK_SPACE_GB=10
CLEANUP_SCHEDULE_HOURS=24
```

## Development

### Running Individual Services

Each service can be run independently for development:

```bash
cd services/api-gateway
pip install -r requirements.txt
python -m app.main
```

### Running Tests

```bash
python -m pytest tests/
```

## Monitoring

- RabbitMQ Management Interface: `http://localhost:15672`
- Service logs are available in the container logs

## Data Retention

- By default, data files are kept for 7 days
- Emergency cleanup triggers when free space falls below 10GB
- Active job files are protected from cleanup

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.