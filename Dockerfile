# 1. Base Image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements first
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 5. Install extra libraries explicitly (as per your request)
RUN pip install --default-timeout=1000 --no-cache-dir streamlit plotly fpdf pytz

# 6. Copy the rest of the application
COPY . .

# 7. Run Command (Fixed for 'frontend/app.py')
# We use ["sh", "-c", "..."] to safely run two commands at once on Render.
# Note the path: 'frontend/app.py'
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0"]