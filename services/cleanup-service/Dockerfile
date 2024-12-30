FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directory for logs
RUN mkdir -p /app/logs

CMD ["python", "-m", "app.main"]