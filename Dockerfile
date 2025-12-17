FROM python:3.10-slim

WORKDIR /app

# Install system tools for swap space
RUN apt-get update && apt-get install -y util-linux

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make script executable
RUN chmod +x entrypoint.sh

# Run the special entrypoint
CMD ["./entrypoint.sh"]