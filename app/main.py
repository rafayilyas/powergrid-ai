from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import pandas as pd
from io import StringIO
from app.schemas import DemandRequest, PeakRequest
from app.utils import load_models

app = FastAPI(title="Electricity Demand Prediction")

# Load models on startup
models = load_models()

@app.post("/predict-demand")
def predict_demand(req: DemandRequest):
    model = models.get('regression')
    if model is None:
        raise HTTPException(status_code=500, detail="Regression model not loaded")
    
    # Inputs must match training: hour, temperature, voltage, dayofweek
    X = [[req.hour, req.temperature, req.voltage, req.dayofweek]]
    
    pred = model.predict(X)[0]
    return {"predicted_demand": float(pred)}

@app.post("/peak-hour")
def peak_hour(req: PeakRequest):
    clf = models.get('classifier')
    if clf is None:
        raise HTTPException(status_code=500, detail="Classifier not loaded")
    
    X = [[req.hour, req.temperature, req.voltage, req.dayofweek]]
    
    # Prediction is 0, 1, or 2
    y = clf.predict(X)[0]
    
    # Map to strings
    labels = {
        0: "Low Risk",
        1: "Moderate Risk",
        2: "High Load Shedding Risk"
    }
    
    return {"risk": labels.get(y, "Unknown")}

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """
    Handles Batch CSV processing.
    Expects CSV with columns: 'hour', 'temperature', 'voltage', 'dayofweek'
    """
    model = models.get('regression')
    if model is None:
        raise HTTPException(status_code=500, detail="Regression model not loaded")

    try:
        # 1. Read the CSV file content
        contents = await file.read()
        csv_string = contents.decode('utf-8')
        
        # 2. Convert to DataFrame
        df = pd.read_csv(StringIO(csv_string))
        
        # 3. Clean headers (remove spaces, lowercase)
        df.columns = df.columns.str.strip().str.lower()
        
        # 4. Check for required columns
        required_cols = ['hour', 'temperature', 'voltage', 'dayofweek']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV is missing required columns: {missing}. Found: {list(df.columns)}"
            )

        # 5. Select ONLY the features the model expects (in correct order)
        # This prevents errors if the CSV has extra columns like 'id' or 'date'
        X = df[required_cols]
        
        # 6. Predict
        predictions = model.predict(X)
        
        return {"predictions": predictions.tolist()}

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")