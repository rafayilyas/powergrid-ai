#!/bin/bash

# 1. Create a 1GB swap file (Fake RAM)
# This prevents the "Out of Memory" crash by using disk space as memory.
fallocate -l 1G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo "Swap enabled"

# 2. Run the App
streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false