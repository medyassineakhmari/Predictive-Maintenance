from fastapi.testclient import TestClient
from main import app

# Création d'un client de test qui simule les requêtes web
client = TestClient(app)

def test_read_root():
    """Vérifie que l'API répond bien sur la route principale."""
    response = client.get("/")
    assert response.status_code == 200
    assert "API de Maintenance Prédictive" in response.json()["message"]

def test_predict_normal_machine():
    """Vérifie que le modèle arrive à prédire une situation normale sans crasher."""
    payload = {
        "Type": 1,
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction_panne" in data
    assert "probabilite" in data
    assert type(data["prediction_panne"]) == bool