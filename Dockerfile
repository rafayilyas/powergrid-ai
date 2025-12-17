# 1. Base Image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements first
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 5. Install extra libraries explicitly
RUN pip install --default-timeout=1000 --no-cache-dir streamlit plotly fpdf pytz

# 6. Copy the rest of the application
COPY . .

# 7. Run Command (UPDATED)
# Added flags: --server.enableCORS=false --server.enableXsrfProtection=false
# This allows the Streamlit frontend to connect via Render's public URL without blocking.
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & \ streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false