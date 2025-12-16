"""
Script de evaluare pentru rețeaua neuronală Text-to-Blender.
==============================================================

Acest script:
1. Încarcă modelul antrenat
2. Evaluează pe setul de test
3. Generează raport detaliat per clasă
4. Calculează metrici suplimentare

Rulare:
    python -m src.neural_network.evaluate
"""

import os
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score,
    top_k_accuracy_score
)
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Adăugare path pentru import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.neural_network.model import NeuralNetwork


def load_model(models_dir: str, config: dict, input_size: int, output_size: int) -> NeuralNetwork:
    """Încarcă modelul antrenat."""
    model = NeuralNetwork(
        input_size=input_size,
        hidden_layers=config.get('model', {}).get('hidden_layers', [128, 64, 32]),
        output_size=output_size,
        activation=config.get('model', {}).get('activation', 'relu'),
        output_activation='softmax',
        dropout=config.get('model', {}).get('dropout', 0.2)
    )
    
    model_path = Path(models_dir) / "trained_model.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelul nu există: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model încărcat din: {model_path}")
    return model


def load_test_data(data_dir: str) -> tuple:
    """Încarcă datele de test."""
    data_path = Path(data_dir) / "test"
    
    X_test = np.load(data_path / "X_test.npy")
    y_test = np.load(data_path / "y_test.npy")
    
    print(f"✅ Date test: {X_test.shape[0]} samples")
    return X_test, y_test


def load_params(config_dir: str) -> dict:
    """Încarcă parametrii de preprocesare."""
    params_path = Path(config_dir) / "preprocessing_params.pkl"
    with open(params_path, 'rb') as f:
        return pickle.load(f)


def load_config(config_dir: str) -> dict:
    """Încarcă configurația modelului."""
    config_path = Path(config_dir) / "model_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def predict(model: NeuralNetwork, X: np.ndarray) -> tuple:
    """Generează predicții."""
    model.eval()
    X_tensor = torch.FloatTensor(X)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()
        predictions = np.argmax(probabilities, axis=1)
    
    return predictions, probabilities


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict:
    """Calculează toate metricile de evaluare."""
    metrics = {
        'accuracy': np.mean(y_true == y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Top-k accuracy cu toate clasele
    num_classes = probabilities.shape[1]
    all_labels = list(range(num_classes))
    
    for k in [3, 5]:
        if k <= num_classes:
            try:
                metrics[f'top_{k}_accuracy'] = top_k_accuracy_score(
                    y_true, probabilities, k=k, labels=all_labels
                )
            except Exception:
                # Fallback dacă top-k nu funcționează
                pass
    
    return metrics


def generate_per_class_report(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    idx_to_intent: dict,
    results_dir: str
):
    """Generează raport detaliat per clasă."""
    results_path = Path(results_dir)
    
    # Găsește clasele prezente în date
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    target_names = [idx_to_intent.get(int(i), f'Class_{i}') for i in unique_labels]
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        labels=unique_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    # Conversie în DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    # Salvare CSV
    report_path = results_path / "per_class_metrics.csv"
    report_df.to_csv(report_path)
    print(f"📊 Raport per clasă salvat: {report_path}")
    
    return report_df


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_intent: dict,
    results_dir: str
):
    """Analizează erorile de clasificare."""
    results_path = Path(results_dir)
    
    # Găsire erori
    errors_mask = y_true != y_pred
    error_indices = np.where(errors_mask)[0]
    
    error_analysis = []
    for idx in error_indices:
        error_analysis.append({
            'sample_idx': int(idx),
            'true_label': idx_to_intent.get(int(y_true[idx]), f'Class_{y_true[idx]}'),
            'predicted_label': idx_to_intent.get(int(y_pred[idx]), f'Class_{y_pred[idx]}')
        })
    
    # Salvare
    errors_df = pd.DataFrame(error_analysis)
    errors_path = results_path / "error_analysis.csv"
    errors_df.to_csv(errors_path, index=False)
    print(f"📊 Analiză erori salvată: {errors_path}")
    
    # Statistici erori
    print(f"\n🔍 Analiză erori:")
    print(f"   Total erori: {len(error_analysis)} din {len(y_true)} ({len(error_analysis)/len(y_true)*100:.2f}%)")
    
    # Cele mai frecvente confuzii
    if len(errors_df) > 0:
        confusion_pairs = errors_df.groupby(['true_label', 'predicted_label']).size().sort_values(ascending=False)
        print(f"\n   Top 5 confuzii:")
        for i, ((true_l, pred_l), count) in enumerate(confusion_pairs.head(5).items()):
            print(f"   {i+1}. {true_l} → {pred_l}: {count} erori")


def plot_class_distribution(y_test: np.ndarray, idx_to_intent: dict, results_dir: str):
    """Generează grafic distribuție clase în test."""
    results_path = Path(results_dir)
    
    # Contorizare
    unique, counts = np.unique(y_test, return_counts=True)
    class_names = [idx_to_intent.get(i, f'Class_{i}')[:20] for i in unique]
    
    # Sortare descrescătoare
    sorted_idx = np.argsort(counts)[::-1][:20]  # Top 20
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(sorted_idx)), counts[sorted_idx], color='steelblue')
    ax.set_xticks(range(len(sorted_idx)))
    ax.set_xticklabels([class_names[i] for i in sorted_idx], rotation=45, ha='right')
    ax.set_xlabel('Clasă')
    ax.set_ylabel('Număr samples')
    ax.set_title('Distribuția claselor în setul de test (Top 20)')
    
    plt.tight_layout()
    plt.savefig(results_path / "test_class_distribution.png", dpi=150)
    plt.close()
    print(f"📊 Distribuție clase salvată: {results_path / 'test_class_distribution.png'}")


def main():
    """Funcția principală de evaluare."""
    print("=" * 60)
    print("📊 Text-to-Blender Model Evaluation")
    print("=" * 60)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    config_dir = project_root / "config"
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    
    # 1. Încărcare parametri și config
    params = load_params(str(config_dir))
    config = load_config(str(config_dir))
    
    # 2. Încărcare date test
    X_test, y_test = load_test_data(str(data_dir))
    
    # 3. Încărcare model
    model = load_model(
        str(models_dir), config,
        input_size=X_test.shape[1],
        output_size=params['num_classes']
    )
    
    # 4. Predicții
    print("\n🔮 Generare predicții...")
    predictions, probabilities = predict(model, X_test)
    
    # 5. Calculare metrici
    print("\n📈 Calculare metrici...")
    metrics = calculate_metrics(y_test, predictions, probabilities)
    
    print(f"\n{'='*40}")
    print("📊 REZULTATE EVALUARE")
    print(f"{'='*40}")
    print(f"  Accuracy:           {metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score (macro):   {metrics['f1_macro']:.4f}")
    print(f"  F1 Score (weighted):{metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    if 'top_3_accuracy' in metrics:
        print(f"  Top-3 Accuracy:     {metrics['top_3_accuracy']*100:.2f}%")
    if 'top_5_accuracy' in metrics:
        print(f"  Top-5 Accuracy:     {metrics['top_5_accuracy']*100:.2f}%")
    print(f"{'='*40}")
    
    # Verificare obiective
    print(f"\n🎯 Verificare obiective Etapa 5:")
    acc_ok = metrics['accuracy'] >= 0.65
    f1_ok = metrics['f1_macro'] >= 0.60
    print(f"   Accuracy ≥ 65%: {'✅ DA' if acc_ok else '❌ NU'} ({metrics['accuracy']*100:.2f}%)")
    print(f"   F1 ≥ 0.60:      {'✅ DA' if f1_ok else '❌ NU'} ({metrics['f1_macro']:.4f})")
    
    # 6. Raport per clasă
    generate_per_class_report(
        y_test, predictions, params['idx_to_intent'], str(results_dir)
    )
    
    # 7. Analiză erori
    analyze_errors(
        y_test, predictions, params['idx_to_intent'], str(results_dir)
    )
    
    # 8. Distribuție clase
    plot_class_distribution(y_test, params['idx_to_intent'], str(results_dir))
    
    # 9. Salvare metrici complete
    eval_metrics_path = results_dir / "evaluation_metrics.json"
    with open(eval_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n📊 Metrici complete salvate: {eval_metrics_path}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluare completă!")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    main()
