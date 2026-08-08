# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="API de prédiction de la dépense calorique",
    description="Prédit les calories brûlées lors d'une activité physique",
    version="1.0"
)

# Charger le modèle une seule fois au démarrage
model = joblib.load("model.pkl")


class ActivityInput(BaseModel):
    Gender: int = Field(..., ge=0, le=1, description="0 = Femme, 1 = Homme")
    Age: int = Field(..., gt=0, description="Âge en années")
    Height: float = Field(..., description="Taille en cm")
    Weight: float = Field(..., description="Poids en kg")
    Duration: float = Field(..., description="Durée de l'exercice en minutes")
    Heart_Rate: float = Field(..., description="Fréquence cardiaque en bpm")
    Body_Temp: float = Field(..., description="Température corporelle en °C")

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": 1,
                "Age": 30,
                "Height": 170,
                "Weight": 70,
                "Duration": 20,
                "Heart_Rate": 100,
                "Body_Temp": 39
            }
        }


@app.get("/health")
def health_check():
    return {"status": "OK"}


@app.post("/predict")
def predict_calories(data: ActivityInput):
    X_new = pd.DataFrame([{
        "const": 1,
        "Gender": data.Gender,
        "Age": data.Age,
        "Height": data.Height,
        "Weight": data.Weight,
        "Duration": data.Duration,
        "Heart_Rate": data.Heart_Rate,
        "Body_Temp": data.Body_Temp,
    }])

    prediction = model.predict(X_new)[0]

    return {
        "calories_predicted": round(float(prediction), 2)
    }