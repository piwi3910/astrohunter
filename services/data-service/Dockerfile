FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directory for SQLite database
RUN mkdir -p /app/data/db
RUN chmod 777 /app/data/db

CMD ["python", "-m", "app.main"]