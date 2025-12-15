# 1. Base Image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements first
COPY requirements.txt .

# 4. Install dependencies (Cached Layer)
# Docker will reuse the heavy libraries (pandas, numpy) from the previous build
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 4.5. NEW LAYER: Install ALL Frontend libraries explicitly
# UPDATED: Added plotly, fpdf, and pytz so the new features work!
RUN pip install --default-timeout=1000 --no-cache-dir streamlit plotly fpdf pytz

# 5. Copy the rest of the app
COPY . /app

# 6. Expose ports
EXPOSE 8000
EXPOSE 8501

# 7. Command to run BOTH services
# Backend: app.main:app | Frontend: frontend/app.py
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0