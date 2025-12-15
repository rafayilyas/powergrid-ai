from pydantic import BaseModel

class DemandRequest(BaseModel):
    hour: int
    temperature: float
    voltage: float
    dayofweek: int  # Added to match training data

class PeakRequest(BaseModel):
    hour: int
    temperature: float
    voltage: float
    dayofweek: int  # Added to match training data