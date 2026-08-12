from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Initialisation de l'API
app = FastAPI(
    title="Predictive Maintenance API",
    description="API MLOps pour la prédiction des pannes machines en temps réel.",
    version="1.0.0"
)

# Constantes (à adapter selon l'emplacement de tes fichiers)
MODEL_PATH = "random_forest_model.joblib"
SCALER_PATH = "standard_scaler.joblib"
SEUIL_OPTIMAL = 0.3

# Chargement du modèle et du scaler au démarrage de l'API
try:
    rf_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Modèle et Scaler chargés avec succès.")
except Exception as e:
    rf_model, scaler = None, None
    print(f"Attention au lancement : {e}")

# Définition du schéma de données attendu (Pydantic)
class MachineData(BaseModel):
    Type: int = Field(..., description="0 pour L (Low), 1 pour M (Medium), 2 pour H (High)")
    Air_temperature_K: float = Field(..., alias="Air temperature [K]")
    Process_temperature_K: float = Field(..., alias="Process temperature [K]")
    Rotational_speed_rpm: float = Field(..., alias="Rotational speed [rpm]")
    Torque_Nm: float = Field(..., alias="Torque [Nm]")
    Tool_wear_min: float = Field(..., alias="Tool wear [min]")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Type": 0,
                "Air temperature [K]": 298.9,
                "Process temperature [K]": 309.0,
                "Rotational speed [rpm]": 1410.0,
                "Torque [Nm]": 65.7,
                "Tool wear [min]": 191.0
            }
        }

@app.get("/")
def read_root():
    return {"message": "API de Maintenance Prédictive active. Accédez à /docs pour l'interface de test."}

@app.post("/predict")
def predict_failure(data: MachineData):
    if rf_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Les artefacts ML ne sont pas chargés sur le serveur.")

    # 1. Conversion des données entrantes en dictionnaire (en respectant les alias/noms de colonnes d'origine)
    input_data = data.model_dump(by_alias=True)
    
    # 2. Création du DataFrame avec exactement les mêmes colonnes que lors de l'entraînement
    expected_cols = [
        'Type', 
        'Air temperature [K]', 
        'Process temperature [K]', 
        'Rotational speed [rpm]', 
        'Torque [Nm]', 
        'Tool wear [min]'
    ]
    df_input = pd.DataFrame([input_data], columns=expected_cols)

    # 3. Standardisation avec le scaler figé
    scaled_features = scaler.transform(df_input)

    # 4. Prédiction de la probabilité de panne (classe 1)
    prob = rf_model.predict_proba(scaled_features)[0, 1]

    # 5. Application de ta logique métier et du seuil optimal
    is_failure = bool(prob >= SEUIL_OPTIMAL)

    if prob >= 0.7:
        status = "CRITIQUE - Arrêt immédiat recommandé"
    elif prob >= SEUIL_OPTIMAL:
        status = "ALERTE - Inspection requise"
    else:
        status = "NORMAL"

    # 6. Réponse JSON de l'API
    return {
        "prediction_panne": is_failure,
        "probabilite": round(prob, 4),
        "seuil_applique": SEUIL_OPTIMAL,
        "statut_recommande": status
    }
