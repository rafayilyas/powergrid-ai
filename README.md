# Electricity Load Shedding & Power Demand Prediction

This repository contains an end-to-end MLOps project for predicting electricity demand, identifying peak hours, and flagging load-shedding risk.

See the `app/`, `training/`, `prefect/`, and `tests/` folders for implementation details.

Run the local sample flow and API using Docker or Python directly.

Dataset:

The primary dataset used for experiments is the UCI "Individual household electric power consumption" dataset. Download the file and save it as `data/raw/household_power_consumption.txt` (it's semicolon-separated and uses `?` for missing values).

UCI dataset link: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Quick start:

1. Create and activate virtualenv:

	- Windows (PowerShell):
	  ```powershell
	  python -m venv .venv
	  .\.venv\Scripts\Activate.ps1
	  pip install -r requirements.txt
	  ```

2. If you want to experiment with a synthetic dataset instead of downloading, generate UCI-like sample data and run training (or use Prefect):

	```powershell
	python data/download_uci.py  # downloads and extracts UCI dataset to data/raw
	python data/generate_sample_data.py  # alternatively, create a small UCI-like sample
	python -m training.preprocess
	python -m training.train_regression
	python -m training.train_classification
	python -m training.train_timeseries
	```

3. Run FastAPI locally:

	```powershell
	uvicorn app.main:app --reload --port 8000
	```

