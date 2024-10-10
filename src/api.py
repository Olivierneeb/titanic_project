from pathlib import Path
from fastapi import FastAPI
import joblib
import pandas as pd
from preprocess import preprocess_data

app = FastAPI()

project_root = Path(__file__).resolve().parent.parent
model_path = project_root / 'models' / 'titanic_model.pkl'
model = joblib.load(model_path)

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(preprocess_data(df))
    return {"prediction": int(prediction[0])}

