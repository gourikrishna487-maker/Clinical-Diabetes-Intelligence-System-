from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import sqlite3
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import DiabetesPredictor
from recommendation import DiabetesRecommendationEngine

app = FastAPI(title="Bio-Forecast Pro API")

# DB Path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database', 'db.sqlite3')

class ClinicalInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

class LifestyleInput(BaseModel):
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI: float
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int
    AnyHealthcare: int
    NoDocbcCost: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Sex: int
    Age: int
    Education: int
    Income: int

# --- API ENDPOINTS ---

@app.post("/api/predict/clinical")
def predict_clinical(data: ClinicalInput):
    try:
        predictor = DiabetesPredictor(mode='clinical')
        engine = DiabetesRecommendationEngine()
        input_dict = data.model_dump()
        result = predictor.predict(input_dict)
        # Add SHAP explanations
        shap_values = predictor.get_explanations(result['input_df'])
        recs = engine.get_recommendations(result['probability'], input_dict)
        return {
            "probability": result['probability'],
            "risk_level": recs['risk_level'],
            "advice": recs['advice'],
            "prediction": result['prediction_class'],
            "shap": shap_values
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/lifestyle")
def predict_lifestyle(data: LifestyleInput):
    try:
        predictor = DiabetesPredictor(mode='lifestyle')
        engine = DiabetesRecommendationEngine()
        input_dict = data.model_dump()
        result = predictor.predict(input_dict)
        # Add SHAP explanations
        shap_values = predictor.get_explanations(result['input_df'])
        recs = engine.get_recommendations(result['probability'], input_dict)
        return {
            "probability": result['probability'],
            "risk_level": recs['risk_level'],
            "advice": recs['advice'],
            "prediction": result['prediction_class'],
            "shap": shap_values
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
def get_history():
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT 20")
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- FRONTEND SERVING ---

@app.get("/", response_class=HTMLResponse)
def get_frontend():
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>Bio-Forecast Pro API is running</h1><p>Frontend file missing.</p>"

if __name__ == "__main__":
    import uvicorn
    # Allow hot reload for development
    uvicorn.run(app, host="0.0.0.0", port=8000)
