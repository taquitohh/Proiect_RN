"""
Script de antrenare pentru rețeaua neuronală Text-to-Blender.
==============================================================

Acest script:
1. Încarcă datele de antrenare și validare
2. Antrenează modelul cu configurația specificată
3. Salvează modelul antrenat și istoricul
4. Generează rapoarte și vizualizări

Rulare:
    python -m src.neural_network.train

Sau din directorul src/neural_network:
    python train.py
"""

import os
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import yaml
import seaborn as sns

# Adăugare path pentru import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.neural_network.model import NeuralNetwork, NeuralNetworkTrainer


def load_data(data_dir: str) -> tuple:
    """
    Încarcă datele de antrenare, validare și test.
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    data_path = Path(data_dir)
    
    # Încărcare date
    X_train = np.load(data_path / "train" / "X_train.npy")
    y_train = np.load(data_path / "train" / "y_train.npy")
    
    X_val = np.load(data_path / "validation" / "X_val.npy")
    y_val = np.load(data_path / "validation" / "y_val.npy")
    
    X_test = np.load(data_path / "test" / "X_test.npy")
    y_test = np.load(data_path / "test" / "y_test.npy")
    
    print(f"✅ Date încărcate:")
    print(f"   Train: {X_train.shape[0]} samples, input_size={X_train.shape[1]}")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_preprocessing_params(config_dir: str) -> dict:
    """Încarcă parametrii de preprocesare."""
    params_path = Path(config_dir) / "preprocessing_params.pkl"
    
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    
    print(f"✅ Parametri preprocesare încărcați:")
    print(f"   Vocabular: {params['vocab_size']} cuvinte")
    print(f"   Intenții: {params['num_classes']} clase")
    
    return params


def load_model_config(config_dir: str) -> dict:
    """Încarcă configurația modelului."""
    config_path = Path(config_dir) / "model_config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def create_model(input_size: int, output_size: int, config: dict) -> NeuralNetwork:
    """Creează modelul neural."""
    model_config = config.get('model', {})
    
    model = NeuralNetwork(
        input_size=input_size,
        hidden_layers=model_config.get('hidden_layers', [128, 64, 32]),
        output_size=output_size,
        activation=model_config.get('activation', 'relu'),
        output_activation='softmax',
        dropout=model_config.get('dropout', 0.2)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model creat:")
    print(f"   Arhitectură: {input_size} → {model_config.get('hidden_layers', [128, 64, 32])} → {output_size}")
    print(f"   Parametri totali: {total_params:,}")
    print(f"   Parametri antrenabili: {trainable_params:,}")
    
    return model


def train_model(
    model: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict
) -> tuple:
    """
    Antrenează modelul.
    
    Returns:
        tuple: (trainer, history)
    """
    print(f"\n🚀 Începere antrenare...")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    training_config = config.get('training', {})
    
    # Actualizare config pentru trainer
    trainer_config = {
        'training': {
            'epochs': training_config.get('epochs', 100),
            'batch_size': training_config.get('batch_size', 32),
            'learning_rate': training_config.get('learning_rate', 0.001),
            'optimizer': training_config.get('optimizer', 'adam'),
            'early_stopping': {
                'enabled': training_config.get('early_stopping', {}).get('enabled', True),
                'patience': training_config.get('early_stopping', {}).get('patience', 15),
                'min_delta': training_config.get('early_stopping', {}).get('min_delta', 0.001)
            }
        },
        'model': {
            'problem_type': 'classification'
        }
    }
    
    print(f"   Epochs: {trainer_config['training']['epochs']}")
    print(f"   Batch size: {trainer_config['training']['batch_size']}")
    print(f"   Learning rate: {trainer_config['training']['learning_rate']}")
    print(f"   Optimizer: {trainer_config['training']['optimizer']}")
    print(f"   Early stopping: patience={trainer_config['training']['early_stopping']['patience']}")
    print("-" * 60)
    
    trainer = NeuralNetworkTrainer(model, trainer_config)
    
    # Antrenare
    start_time = datetime.now()
    history = trainer.train(X_train, y_train, X_val, y_val)
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()
    
    print("-" * 60)
    print(f"✅ Antrenare finalizată în {training_time:.2f} secunde")
    print(f"   Epoci completate: {len(history['train_loss'])}")
    print(f"   Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final Train Acc:  {history['train_acc'][-1]:.4f}")
    
    if history['val_loss']:
        print(f"   Final Val Loss:   {history['val_loss'][-1]:.4f}")
        print(f"   Final Val Acc:    {history['val_acc'][-1]:.4f}")
    
    return trainer, history, training_time


def evaluate_model(
    trainer: NeuralNetworkTrainer,
    X_test: np.ndarray,
    y_test: np.ndarray,
    intent_to_idx: dict
) -> dict:
    """
    Evaluează modelul pe setul de test.
    
    Returns:
        dict: Metrici de evaluare
    """
    print(f"\n📈 Evaluare pe setul de test ({X_test.shape[0]} samples)...")
    
    # Evaluare de bază
    results = trainer.evaluate(X_test, y_test)
    
    predictions = np.array(results['predictions'])
    labels = np.array(results['labels'])
    
    # Calculare F1 score
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    metrics = {
        'test_loss': results['test_loss'],
        'accuracy': results['accuracy'],
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'num_test_samples': len(y_test),
        'num_classes': len(intent_to_idx)
    }
    
    print(f"\n📊 Rezultate evaluare:")
    print(f"   Test Loss:    {metrics['test_loss']:.4f}")
    print(f"   Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   F1 (macro):   {metrics['f1_macro']:.4f}")
    print(f"   F1 (weighted):{metrics['f1_weighted']:.4f}")
    
    # Verificare obiective
    print(f"\n🎯 Verificare obiective Etapa 5:")
    acc_ok = metrics['accuracy'] >= 0.65
    f1_ok = metrics['f1_macro'] >= 0.60
    
    print(f"   Accuracy ≥ 65%: {'✅ DA' if acc_ok else '❌ NU'} ({metrics['accuracy']*100:.2f}%)")
    print(f"   F1 ≥ 0.60:      {'✅ DA' if f1_ok else '❌ NU'} ({metrics['f1_macro']:.4f})")
    
    return metrics, predictions, labels


def save_results(
    trainer: NeuralNetworkTrainer,
    history: dict,
    metrics: dict,
    training_time: float,
    models_dir: str,
    results_dir: str,
    config: dict
):
    """Salvează toate rezultatele antrenării."""
    models_path = Path(models_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Salvare model antrenat
    model_path = models_path / "trained_model.pt"
    trainer.save_model(str(model_path))
    print(f"\n💾 Model salvat: {model_path}")
    
    # 2. Salvare istoric antrenare CSV
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'] if history['val_loss'] else [None] * len(history['train_loss']),
        'val_acc': history['val_acc'] if history['val_acc'] else [None] * len(history['train_loss'])
    })
    history_path = results_path / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"📊 Istoric salvat: {history_path}")
    
    # 3. Salvare metrici JSON
    metrics_full = {
        **metrics,
        'training_time_seconds': training_time,
        'epochs_completed': len(history['train_loss']),
        'final_train_loss': history['train_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        'final_val_acc': history['val_acc'][-1] if history['val_acc'] else None,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': config.get('training', {}).get('epochs', 100),
            'batch_size': config.get('training', {}).get('batch_size', 32),
            'learning_rate': config.get('training', {}).get('learning_rate', 0.001),
            'hidden_layers': config.get('model', {}).get('hidden_layers', [128, 64, 32]),
            'dropout': config.get('model', {}).get('dropout', 0.2)
        }
    }
    
    metrics_path = results_path / "test_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_full, f, indent=2, ensure_ascii=False)
    print(f"📈 Metrici salvate: {metrics_path}")


def plot_training_history(history: dict, results_dir: str):
    """Generează grafice pentru istoricul antrenării."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Creare figură cu 2 subploturi
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Training & Validation Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    if history['val_acc']:
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].axhline(y=0.65, color='g', linestyle='--', label='Target (65%)', alpha=0.7)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvare
    plot_path = results_path / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Grafic salvat: {plot_path}")


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    intent_to_idx: dict,
    results_dir: str,
    top_n: int = 20
):
    """Generează matricea de confuzie (top N clase)."""
    results_path = Path(results_dir)
    
    # Selectăm doar top N clase după frecvență
    unique_labels, counts = np.unique(labels, return_counts=True)
    top_indices = np.argsort(counts)[-top_n:]
    
    # Mapare inversă
    idx_to_intent = {v: k for k, v in intent_to_idx.items()}
    
    # Filtrare pentru top N
    mask = np.isin(labels, unique_labels[top_indices])
    filtered_labels = labels[mask]
    filtered_preds = predictions[mask]
    
    # Creare matricea de confuzie
    cm = confusion_matrix(filtered_labels, filtered_preds, labels=unique_labels[top_indices])
    
    # Nume clase pentru afișare
    class_names = [idx_to_intent.get(idx, f'Class_{idx}')[:15] for idx in unique_labels[top_indices]]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    ax.set_title(f'Confusion Matrix (Top {top_n} Classes)', fontsize=14)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Salvare
    cm_path = results_path / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix salvată: {cm_path}")


def main():
    """Funcția principală de antrenare."""
    print("=" * 60)
    print("🧠 Text-to-Blender Neural Network Training")
    print("=" * 60)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    config_dir = project_root / "config"
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    
    # 1. Încărcare date
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(str(data_dir))
    
    # 2. Încărcare parametri preprocesare
    params = load_preprocessing_params(str(config_dir))
    
    # 3. Încărcare configurație model
    config = load_model_config(str(config_dir))
    
    # 4. Creare model
    input_size = X_train.shape[1]
    output_size = params['num_classes']
    model = create_model(input_size, output_size, config)
    
    # 5. Antrenare
    trainer, history, training_time = train_model(
        model, X_train, y_train, X_val, y_val, config
    )
    
    # 6. Evaluare
    metrics, predictions, labels = evaluate_model(
        trainer, X_test, y_test, params['intent_to_idx']
    )
    
    # 7. Salvare rezultate
    save_results(
        trainer, history, metrics, training_time,
        str(models_dir), str(results_dir), config
    )
    
    # 8. Generare grafice
    plot_training_history(history, str(results_dir))
    plot_confusion_matrix(
        labels, predictions, params['intent_to_idx'], str(results_dir)
    )
    
    print("\n" + "=" * 60)
    print("✅ Antrenare completă!")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    main()
