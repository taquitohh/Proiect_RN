"""
API Backend pentru proiectul de Rețele Neuronale.
=================================================

Acest modul expune un API REST pentru:
- Încărcarea și preprocesarea datelor
- Antrenarea modelului
- Predicții și evaluare
"""

import os
import sys
from pathlib import Path

# Adaugă src în path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import yaml

# Import module locale
from data_acquisition.data_loader import (
    load_csv_data, 
    generate_synthetic_data, 
    get_data_info,
    save_data
)
from preprocessing.preprocessor import DataPreprocessor, save_splits_to_files
from neural_network.model import NeuralNetwork, NeuralNetworkTrainer

# Inițializare aplicație
app = FastAPI(
    title="Neural Network API",
    description="API pentru antrenarea și utilizarea rețelelor neuronale",
    version="1.0.0"
)

# Configurare CORS pentru React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stare globală
app_state = {
    "data": None,
    "preprocessor": None,
    "model": None,
    "trainer": None,
    "splits": None,
    "config": None
}


# ==================== Modele Pydantic ====================

class SyntheticDataRequest(BaseModel):
    n_samples: int = 1000
    n_features: int = 10
    n_classes: int = 3
    noise: float = 0.1


class PreprocessRequest(BaseModel):
    target_column: str
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    normalize: bool = True
    handle_missing: bool = True


class TrainRequest(BaseModel):
    hidden_layers: List[int] = [128, 64, 32]
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    dropout: float = 0.2


class PredictRequest(BaseModel):
    data: List[List[float]]


# ==================== Endpoints Date ====================

@app.get("/")
async def root():
    """Endpoint principal."""
    return {
        "message": "Neural Network API",
        "version": "1.0.0",
        "endpoints": {
            "data": "/api/data/*",
            "preprocess": "/api/preprocess/*",
            "train": "/api/train/*",
            "predict": "/api/predict"
        }
    }


@app.get("/api/status")
async def get_status():
    """Returnează starea curentă a aplicației."""
    return {
        "data_loaded": app_state["data"] is not None,
        "data_shape": app_state["data"].shape if app_state["data"] is not None else None,
        "preprocessed": app_state["splits"] is not None,
        "model_trained": app_state["trainer"] is not None and len(app_state["trainer"].history.get("train_loss", [])) > 0
    }


@app.post("/api/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Încarcă un fișier CSV cu date."""
    try:
        content = await file.read()
        
        # Salvare temporară
        temp_path = f"../data/raw/{file.filename}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Încărcare ca DataFrame
        app_state["data"] = pd.read_csv(temp_path)
        
        info = get_data_info(app_state["data"])
        return {
            "message": f"Fișier încărcat: {file.filename}",
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/data/generate")
async def generate_data(request: SyntheticDataRequest):
    """Generează date sintetice pentru testare."""
    try:
        X, y = generate_synthetic_data(
            n_samples=request.n_samples,
            n_features=request.n_features,
            n_classes=request.n_classes,
            noise=request.noise
        )
        
        app_state["data"] = pd.concat([X, y], axis=1)
        
        # Salvare
        save_data(app_state["data"], "../data/raw/synthetic_data.csv")
        
        info = get_data_info(app_state["data"])
        return {
            "message": "Date sintetice generate",
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/data/info")
async def get_data_info_endpoint():
    """Returnează informații despre datele încărcate."""
    if app_state["data"] is None:
        raise HTTPException(status_code=400, detail="Nu sunt date încărcate")
    
    info = get_data_info(app_state["data"])
    
    # Adaugă primele rânduri
    info["preview"] = app_state["data"].head(10).to_dict(orient="records")
    
    return info


@app.get("/api/data/statistics")
async def get_statistics():
    """Returnează statistici despre date."""
    if app_state["data"] is None:
        raise HTTPException(status_code=400, detail="Nu sunt date încărcate")
    
    data = app_state["data"]
    
    # Statistici pentru coloane numerice
    numeric_stats = data.describe().to_dict()
    
    # Valori lipsă
    missing = data.isnull().sum().to_dict()
    
    return {
        "statistics": numeric_stats,
        "missing_values": missing,
        "dtypes": data.dtypes.astype(str).to_dict()
    }


# ==================== Endpoints Preprocesare ====================

@app.post("/api/preprocess")
async def preprocess_data(request: PreprocessRequest):
    """Preprocesează datele."""
    if app_state["data"] is None:
        raise HTTPException(status_code=400, detail="Nu sunt date încărcate")
    
    try:
        # Creare preprocessor cu configurație
        config = {
            'splitting': {
                'train_ratio': request.train_ratio,
                'validation_ratio': request.validation_ratio,
                'test_ratio': request.test_ratio,
                'random_seed': 42,
                'stratify': True
            },
            'preprocessing': {
                'normalization': {'enabled': request.normalize, 'method': 'minmax'},
                'missing_values': {'strategy': 'median', 'threshold': 0.3},
                'outliers': {'enabled': True, 'method': 'iqr', 'iqr_multiplier': 1.5}
            }
        }
        
        preprocessor = DataPreprocessor()
        preprocessor.config = config
        
        result = preprocessor.preprocess_pipeline(
            app_state["data"],
            target_column=request.target_column
        )
        
        app_state["preprocessor"] = preprocessor
        app_state["splits"] = result["splits"]
        
        # Salvare în fișiere
        save_splits_to_files(result["splits"], "../data")
        
        return {
            "message": "Date preprocesate cu succes",
            "train_size": len(result["splits"]["train"][0]),
            "validation_size": len(result["splits"]["validation"][0]),
            "test_size": len(result["splits"]["test"][0]),
            "analysis": {
                "shape": result["analysis"]["shape"],
                "issues": result["analysis"]["issues"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/preprocess/analysis")
async def get_analysis():
    """Returnează analiza datelor."""
    if app_state["data"] is None:
        raise HTTPException(status_code=400, detail="Nu sunt date încărcate")
    
    preprocessor = DataPreprocessor()
    analysis = preprocessor.analyze_data(app_state["data"])
    
    return analysis


# ==================== Endpoints Antrenare ====================

@app.post("/api/train")
async def train_model(request: TrainRequest):
    """Antrenează modelul."""
    if app_state["splits"] is None:
        raise HTTPException(status_code=400, detail="Datele nu sunt preprocesate")
    
    try:
        X_train, y_train = app_state["splits"]["train"]
        X_val, y_val = app_state["splits"]["validation"]
        
        # Conversie la numpy
        X_train_np = X_train.values.astype(np.float32)
        y_train_np = y_train.values
        X_val_np = X_val.values.astype(np.float32)
        y_val_np = y_val.values
        
        # Determinare dimensiuni
        input_size = X_train_np.shape[1]
        output_size = len(np.unique(y_train_np))
        
        # Creare model
        model = NeuralNetwork(
            input_size=input_size,
            hidden_layers=request.hidden_layers,
            output_size=output_size,
            dropout=request.dropout
        )
        
        # Configurare antrenare
        config = {
            'training': {
                'epochs': request.epochs,
                'batch_size': request.batch_size,
                'learning_rate': request.learning_rate,
                'optimizer': 'adam',
                'early_stopping': {
                    'enabled': True,
                    'patience': 10,
                    'min_delta': 0.001
                }
            },
            'model': {
                'problem_type': 'classification'
            }
        }
        
        trainer = NeuralNetworkTrainer(model, config)
        history = trainer.train(X_train_np, y_train_np, X_val_np, y_val_np)
        
        app_state["model"] = model
        app_state["trainer"] = trainer
        
        # Salvare model
        trainer.save_model("../models/model_checkpoint.pth")
        
        return {
            "message": "Model antrenat cu succes",
            "epochs_trained": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/train/history")
async def get_training_history():
    """Returnează istoricul antrenării."""
    if app_state["trainer"] is None:
        raise HTTPException(status_code=400, detail="Modelul nu este antrenat")
    
    return app_state["trainer"].history


@app.get("/api/train/evaluate")
async def evaluate_model():
    """Evaluează modelul pe setul de test."""
    if app_state["trainer"] is None or app_state["splits"] is None:
        raise HTTPException(status_code=400, detail="Modelul nu este antrenat")
    
    try:
        X_test, y_test = app_state["splits"]["test"]
        X_test_np = X_test.values.astype(np.float32)
        y_test_np = y_test.values
        
        results = app_state["trainer"].evaluate(X_test_np, y_test_np)
        
        return {
            "test_loss": results["test_loss"],
            "accuracy": results.get("accuracy"),
            "n_samples": len(y_test)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Endpoints Predicție ====================

@app.post("/api/predict")
async def predict(request: PredictRequest):
    """Realizează predicții cu modelul antrenat."""
    if app_state["model"] is None:
        raise HTTPException(status_code=400, detail="Modelul nu este antrenat")
    
    try:
        import torch
        
        data = np.array(request.data, dtype=np.float32)
        
        # Normalizare dacă preprocessor-ul are scaler
        if app_state["preprocessor"] and app_state["preprocessor"].scalers:
            data = app_state["preprocessor"].scalers["main"].transform(data)
        
        # Predicție
        X_tensor = torch.FloatTensor(data)
        predictions = app_state["model"].predict(X_tensor)
        
        predicted_classes = predictions.argmax(dim=1).numpy().tolist()
        probabilities = predictions.numpy().tolist()
        
        return {
            "predictions": predicted_classes,
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Config ====================

@app.get("/api/config/model")
async def get_model_config():
    """Returnează configurația modelului."""
    config_path = "../config/model_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


@app.get("/api/config/preprocessing")
async def get_preprocessing_config():
    """Returnează configurația de preprocesare."""
    config_path = "../config/preprocessing_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
