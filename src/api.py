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
from flask import Flask, request, jsonify
from flask_cors import CORS
import yaml

# Import module locale (lazy loading pentru torch)
from data_acquisition.data_loader import (
    load_csv_data, 
    generate_synthetic_data, 
    get_data_info,
    save_data
)
from preprocessing.preprocessor import DataPreprocessor, save_splits_to_files

# Neural network se importă lazy pentru a evita erori la startup
NeuralNetwork = None
NeuralNetworkTrainer = None

def load_neural_network_module():
    """Încarcă modulul neural network la cerere."""
    global NeuralNetwork, NeuralNetworkTrainer
    if NeuralNetwork is None:
        from neural_network.model import NeuralNetwork as NN, NeuralNetworkTrainer as NNT
        NeuralNetwork = NN
        NeuralNetworkTrainer = NNT
    return NeuralNetwork, NeuralNetworkTrainer

# Inițializare aplicație
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Configurare CORS pentru React
CORS(app, origins=["http://localhost:3000", "http://localhost:5173"],
     supports_credentials=True)

# Stare globală
app_state: Dict[str, Any] = {
    "data": None,
    "preprocessor": None,
    "model": None,
    "trainer": None,
    "splits": None,
    "config": None
}


# ==================== Helper Functions ====================

def error_response(message: str, status_code: int = 400):
    """Returnează un răspuns de eroare."""
    return jsonify({"error": message}), status_code


# ==================== Endpoints Date ====================

@app.route("/")
def root():
    """Endpoint principal."""
    return jsonify({
        "message": "Neural Network API",
        "version": "1.0.0",
        "endpoints": {
            "data": "/api/data/*",
            "preprocess": "/api/preprocess/*",
            "train": "/api/train/*",
            "predict": "/api/predict"
        }
    })


@app.route("/api/status")
def get_status():
    """Returnează starea curentă a aplicației."""
    return jsonify({
        "data_loaded": app_state["data"] is not None,
        "data_shape": list(app_state["data"].shape) if app_state["data"] is not None else None,
        "preprocessed": app_state["splits"] is not None,
        "model_trained": app_state["trainer"] is not None and len(app_state["trainer"].history.get("train_loss", [])) > 0
    })


@app.route("/api/data/upload", methods=["POST"])
def upload_data():
    """Încarcă un fișier CSV cu date."""
    try:
        if 'file' not in request.files:
            return error_response("Nu a fost trimis niciun fișier")
        
        file = request.files['file']
        if file.filename == '':
            return error_response("Numele fișierului este gol")
        
        content = file.read()
        
        # Salvare temporară
        temp_path = f"../data/raw/{file.filename}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Încărcare ca DataFrame
        app_state["data"] = pd.read_csv(temp_path)
        
        info = get_data_info(app_state["data"])
        return jsonify({
            "message": f"Fișier încărcat: {file.filename}",
            "info": info
        })
    except Exception as e:
        return error_response(str(e))


@app.route("/api/data/generate", methods=["POST"])
def generate_data():
    """Generează date sintetice pentru testare."""
    try:
        data = request.get_json() or {}
        n_samples = data.get('n_samples', 1000)
        n_features = data.get('n_features', 10)
        n_classes = data.get('n_classes', 3)
        noise = data.get('noise', 0.1)
        
        X, y = generate_synthetic_data(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            noise=noise
        )
        
        app_state["data"] = pd.concat([X, y], axis=1)
        
        # Salvare
        save_data(app_state["data"], "../data/raw/synthetic_data.csv")
        
        info = get_data_info(app_state["data"])
        return jsonify({
            "message": "Date sintetice generate",
            "info": info
        })
    except Exception as e:
        return error_response(str(e))


@app.route("/api/data/info")
def get_data_info_endpoint():
    """Returnează informații despre datele încărcate."""
    if app_state["data"] is None:
        return error_response("Nu sunt date încărcate")
    
    info = get_data_info(app_state["data"])
    
    # Adaugă primele rânduri
    info["preview"] = app_state["data"].head(10).to_dict(orient="records")
    
    return jsonify(info)


@app.route("/api/data/statistics")
def get_statistics():
    """Returnează statistici despre date."""
    if app_state["data"] is None:
        return error_response("Nu sunt date încărcate")
    
    data = app_state["data"]
    
    # Statistici pentru coloane numerice
    numeric_stats = data.describe().to_dict()
    
    # Valori lipsă
    missing = data.isnull().sum().to_dict()
    
    return jsonify({
        "statistics": numeric_stats,
        "missing_values": missing,
        "dtypes": data.dtypes.astype(str).to_dict()
    })


# ==================== Endpoints Preprocesare ====================

@app.route("/api/preprocess", methods=["POST"])
def preprocess_data():
    """Preprocesează datele."""
    if app_state["data"] is None:
        return error_response("Nu sunt date încărcate")
    
    try:
        data = request.get_json() or {}
        target_column = data.get('target_column')
        if not target_column:
            return error_response("target_column este obligatoriu")
        
        train_ratio = data.get('train_ratio', 0.8)
        validation_ratio = data.get('validation_ratio', 0.1)
        test_ratio = data.get('test_ratio', 0.1)
        normalize = data.get('normalize', True)
        
        # Creare preprocessor cu configurație
        config = {
            'splitting': {
                'train_ratio': train_ratio,
                'validation_ratio': validation_ratio,
                'test_ratio': test_ratio,
                'random_seed': 42,
                'stratify': True
            },
            'preprocessing': {
                'normalization': {'enabled': normalize, 'method': 'minmax'},
                'missing_values': {'strategy': 'median', 'threshold': 0.3},
                'outliers': {'enabled': True, 'method': 'iqr', 'iqr_multiplier': 1.5}
            }
        }
        
        preprocessor = DataPreprocessor()
        preprocessor.config = config
        
        result = preprocessor.preprocess_pipeline(
            app_state["data"],
            target_column=target_column
        )
        
        app_state["preprocessor"] = preprocessor
        app_state["splits"] = result["splits"]
        
        # Salvare în fișiere
        save_splits_to_files(result["splits"], "../data")
        
        return jsonify({
            "message": "Date preprocesate cu succes",
            "train_size": len(result["splits"]["train"][0]),
            "validation_size": len(result["splits"]["validation"][0]),
            "test_size": len(result["splits"]["test"][0]),
            "analysis": {
                "shape": result["analysis"]["shape"],
                "issues": result["analysis"]["issues"]
            }
        })
    except Exception as e:
        return error_response(str(e))


@app.route("/api/preprocess/analysis")
def get_analysis():
    """Returnează analiza datelor."""
    if app_state["data"] is None:
        return error_response("Nu sunt date încărcate")
    
    preprocessor = DataPreprocessor()
    analysis = preprocessor.analyze_data(app_state["data"])
    
    return jsonify(analysis)


# ==================== Endpoints Antrenare ====================

@app.route("/api/train", methods=["POST"])
def train_model():
    """Antrenează modelul."""
    if app_state["splits"] is None:
        return error_response("Datele nu sunt preprocesate")
    
    try:
        data = request.get_json() or {}
        hidden_layers = data.get('hidden_layers', [128, 64, 32])
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 32)
        learning_rate = data.get('learning_rate', 0.001)
        dropout = data.get('dropout', 0.2)
        
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
            hidden_layers=hidden_layers,
            output_size=output_size,
            dropout=dropout
        )
        
        # Configurare antrenare
        config = {
            'training': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
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
        
        return jsonify({
            "message": "Model antrenat cu succes",
            "epochs_trained": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
            "history": history
        })
    except Exception as e:
        return error_response(str(e))


@app.route("/api/train/history")
def get_training_history():
    """Returnează istoricul antrenării."""
    if app_state["trainer"] is None:
        return error_response("Modelul nu este antrenat")
    
    return jsonify(app_state["trainer"].history)


@app.route("/api/train/evaluate")
def evaluate_model():
    """Evaluează modelul pe setul de test."""
    if app_state["trainer"] is None or app_state["splits"] is None:
        return error_response("Modelul nu este antrenat")
    
    try:
        X_test, y_test = app_state["splits"]["test"]
        X_test_np = X_test.values.astype(np.float32)
        y_test_np = y_test.values
        
        results = app_state["trainer"].evaluate(X_test_np, y_test_np)
        
        return jsonify({
            "test_loss": results["test_loss"],
            "accuracy": results.get("accuracy"),
            "n_samples": len(y_test)
        })
    except Exception as e:
        return error_response(str(e))


# ==================== Endpoints Predicție ====================

@app.route("/api/predict", methods=["POST"])
def predict():
    """Realizează predicții cu modelul antrenat."""
    if app_state["model"] is None:
        return error_response("Modelul nu este antrenat")
    
    try:
        import torch
        
        req_data = request.get_json() or {}
        data = req_data.get('data', [])
        
        if not data:
            return error_response("Nu au fost trimise date pentru predicție")
        
        data = np.array(data, dtype=np.float32)
        
        # Normalizare dacă preprocessor-ul are scaler
        if app_state["preprocessor"] and app_state["preprocessor"].scalers:
            data = app_state["preprocessor"].scalers["main"].transform(data)
        
        # Predicție
        X_tensor = torch.FloatTensor(data)
        predictions = app_state["model"].predict(X_tensor)
        
        predicted_classes = predictions.argmax(dim=1).numpy().tolist()
        probabilities = predictions.numpy().tolist()
        
        return jsonify({
            "predictions": predicted_classes,
            "probabilities": probabilities
        })
    except Exception as e:
        return error_response(str(e))


# ==================== Config ====================

@app.route("/api/config/model")
def get_model_config():
    """Returnează configurația modelului."""
    config_path = "../config/model_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return jsonify(yaml.safe_load(f))
    return jsonify({})


@app.route("/api/config/preprocessing")
def get_preprocessing_config():
    """Returnează configurația de preprocesare."""
    config_path = "../config/preprocessing_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return jsonify(yaml.safe_load(f))
    return jsonify({})


# ==================== Blender Text-to-Code ====================

from generators.blender_generator import BlenderScriptGenerator
from neural_network.inference import get_classifier, classify_intent

# Inițializare generator Blender
blender_generator = BlenderScriptGenerator()

# Încărcare clasificator RN la startup
intent_classifier = get_classifier()

# Mapping simplu text → intenție (FALLBACK dacă modelul nu e încărcat)
INTENT_KEYWORDS = {
    "create_cube": ["cub", "cube", "box", "cutie"],
    "create_sphere": ["sfera", "sferă", "sphere", "bila", "bilă", "ball"],
    "create_cylinder": ["cilindru", "cylinder", "tub"],
    "create_cone": ["con", "cone", "piramida", "piramidă"],
    "create_torus": ["torus", "inel", "ring", "gogoasa", "gogoașă"],
    "delete_all": ["sterge", "șterge", "delete", "elimina", "elimină", "curata", "curăță"],
    "apply_material": ["material", "culoare", "color", "textura", "textură"],
}

def classify_intent_simple(text: str) -> str:
    """
    Clasificare simplă bazată pe cuvinte cheie.
    Folosită ca FALLBACK dacă modelul RN nu e disponibil.
    """
    text_lower = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return intent
    return "create_cube"  # Default


def classify_intent_hybrid(text: str) -> dict:
    """
    Clasificare hibridă: folosește RN dacă e disponibil, altfel reguli.
    
    Returns:
        Dict cu intent, confidence, method
    """
    # Încearcă mai întâi cu modelul RN
    if intent_classifier.is_loaded:
        result = intent_classifier.predict(text, top_k=3)
        if result.get("confidence", 0) > 0.3:  # Threshold minim
            return result
    
    # Fallback la reguli simple
    intent = classify_intent_simple(text)
    return {
        "intent": intent,
        "confidence": 0.8,  # Artificial pentru reguli
        "top_k": [{"intent": intent, "confidence": 0.8}],
        "method": "keyword_rules"
    }


@app.route("/api/blender/generate", methods=["POST"])
def generate_blender_code():
    """
    Endpoint principal pentru generarea de cod Blender din text.
    
    Pipeline:
    1. Primește text de la utilizator
    2. Clasifică intenția cu RN antrenat (sau fallback reguli)
    3. Extrage parametrii din text
    4. Generează script Python pentru Blender
    """
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return error_response("Lipsește câmpul 'text' din request")
        
        user_text = data["text"]
        
        # 1. Clasificare intenție cu model RN sau fallback
        classification = classify_intent_hybrid(user_text)
        intent = classification["intent"]
        confidence = classification["confidence"]
        method = classification.get("method", "unknown")
        top_k = classification.get("top_k", [])
        
        # 2. Extragere parametri din text
        params = blender_generator.extract_parameters_from_text(user_text)
        
        # 3. Generare script
        script = blender_generator.generate_script(intent, params)
        
        # 4. Returnare răspuns
        return jsonify({
            "success": True,
            "intent": intent,
            "confidence": confidence,
            "classification_method": method,
            "top_predictions": top_k,
            "params": params,
            "interpretation": f"Am înțeles că vrei să: {intent.replace('_', ' ')} (încredere: {confidence*100:.1f}%)",
            "code": script
        })
        
    except Exception as e:
        return error_response(f"Eroare la generare: {str(e)}")


@app.route("/api/blender/intents")
def get_blender_intents():
    """Returnează lista de intenții disponibile."""
    if intent_classifier.is_loaded:
        intents = intent_classifier.get_available_intents()
        return jsonify({
            "intents": intents,
            "count": len(intents),
            "model_loaded": True,
            "method": "neural_network"
        })
    else:
        return jsonify({
            "intents": list(INTENT_KEYWORDS.keys()),
            "keywords": INTENT_KEYWORDS,
            "model_loaded": False,
            "method": "keyword_rules"
        })


@app.route("/api/blender/model-status")
def get_model_status():
    """Returnează starea modelului de clasificare."""
    return jsonify({
        "model_loaded": intent_classifier.is_loaded,
        "vocab_size": intent_classifier.vocab_size if intent_classifier.is_loaded else 0,
        "num_intents": intent_classifier.num_classes if intent_classifier.is_loaded else len(INTENT_KEYWORDS),
        "method": "neural_network" if intent_classifier.is_loaded else "keyword_rules"
    })


if __name__ == "__main__":
    print("Starting Flask server on http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)
