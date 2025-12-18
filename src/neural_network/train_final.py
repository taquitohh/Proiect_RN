"""
Script de antrenare FINAL cu toate cerințele Nivel 2 și Nivel 3.
================================================================

Caracteristici implementate:
- ✅ Early Stopping (patience=10)
- ✅ Learning Rate Scheduler (ReduceLROnPlateau)
- ✅ Augmentări date NLP (sinonime, swap cuvinte)
- ✅ Weight Decay (L2 regularization)
- ✅ Export ONNX cu benchmark latență
"""

import os
import sys
import json
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class SimpleNeuralNetwork(nn.Module):
    """Rețea neuronală compatibilă cu clasa NeuralNetwork din model.py."""
    
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.3):
        super(SimpleNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# AUGMENTĂRI DATE NLP - Nivel 2
# =============================================================================

# Dicționar sinonime românești pentru comenzi Blender
SINONIME = {
    'creează': ['fă', 'generează', 'adaugă', 'construiește'],
    'fă': ['creează', 'generează', 'adaugă'],
    'șterge': ['elimină', 'remove', 'înlătură'],
    'elimină': ['șterge', 'remove', 'înlătură'],
    'mută': ['deplasează', 'mișcă', 'translatează', 'pune'],
    'deplasează': ['mută', 'mișcă', 'translatează'],
    'rotește': ['întoarce', 'pivotează', 'roteaz'],
    'întoarce': ['rotește', 'pivotează'],
    'scalează': ['redimensionează', 'mărește', 'micșorează'],
    'mărește': ['scalează', 'crește', 'amplifica'],
    'micșorează': ['scalează', 'reduce', 'diminuează'],
    'cub': ['box', 'cutie', 'pătrat'],
    'sferă': ['bilă', 'glob', 'minge'],
    'cilindru': ['tub', 'țeavă'],
    'con': ['piramidă', 'vârf'],
    'plan': ['suprafață', 'podea', 'bază'],
    'obiect': ['element', 'formă', 'corp'],
    'tot': ['toate', 'complet', 'întreg'],
    'roșu': ['red', 'carmin', 'rubiniu'],
    'albastru': ['blue', 'azur', 'ceruleu'],
    'verde': ['green', 'smarald'],
    'mare': ['large', 'big', 'imens'],
    'mic': ['small', 'little', 'minuscul'],
}


def augment_text(text: str, num_augmentations: int = 2) -> list:
    """
    Augmentează textul folosind tehnici NLP.
    
    Tehnici aplicate:
    1. Înlocuire cu sinonime
    2. Swap aleator de cuvinte
    3. Ștergere aleatoare de cuvinte (pentru robustețe)
    """
    augmented = []
    words = text.lower().split()
    
    for _ in range(num_augmentations):
        new_words = words.copy()
        technique = random.choice(['sinonim', 'swap', 'delete'])
        
        if technique == 'sinonim':
            # Înlocuire cu sinonim
            for i, word in enumerate(new_words):
                if word in SINONIME and random.random() < 0.3:
                    new_words[i] = random.choice(SINONIME[word])
        
        elif technique == 'swap':
            # Swap aleator de 2 cuvinte adiacente
            if len(new_words) > 2:
                idx = random.randint(0, len(new_words) - 2)
                new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]
        
        elif technique == 'delete':
            # Ștergere aleatoare (păstrăm minim 2 cuvinte)
            if len(new_words) > 2:
                del_idx = random.randint(0, len(new_words) - 1)
                new_words.pop(del_idx)
        
        augmented_text = ' '.join(new_words)
        if augmented_text != text.lower():
            augmented.append(augmented_text)
    
    return augmented


def augment_dataset(X_train, y_train, vocab, augmentation_factor=0.3):
    """
    Augmentează dataset-ul de antrenare.
    
    Args:
        X_train: Features originale (bag of words)
        y_train: Labels originale
        vocab: Vocabularul pentru reconstrucție text
        augmentation_factor: Proporția de augmentări (0.3 = 30% date noi)
    
    Returns:
        X_augmented, y_augmented: Date augmentate
    """
    print(f"\n📈 Augmentare date NLP...")
    print(f"   Date originale: {len(X_train)} samples")
    
    # Creăm word_to_idx pentru reconstrucție
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}
    
    # Numărul de samples noi de generat
    num_to_augment = int(len(X_train) * augmentation_factor)
    
    # Selectăm random samples pentru augmentare
    indices = np.random.choice(len(X_train), num_to_augment, replace=True)
    
    new_X = []
    new_y = []
    
    for idx in indices:
        # Reconstruim textul din bag of words (aproximativ)
        x = X_train[idx]
        words = []
        for word_idx, count in enumerate(x):
            if count > 0 and word_idx < len(idx_to_word):
                words.extend([idx_to_word[word_idx]] * int(count))
        
        if len(words) < 2:
            continue
            
        original_text = ' '.join(words)
        
        # Augmentăm
        augmented_texts = augment_text(original_text, num_augmentations=1)
        
        for aug_text in augmented_texts:
            # Convertim înapoi la bag of words
            new_bow = np.zeros(len(vocab))
            for word in aug_text.split():
                if word in word_to_idx:
                    new_bow[word_to_idx[word]] += 1
            
            if np.sum(new_bow) > 0:  # Validare că avem cuvinte valide
                new_X.append(new_bow)
                new_y.append(y_train[idx])
    
    if len(new_X) > 0:
        X_augmented = np.vstack([X_train, np.array(new_X)])
        y_augmented = np.concatenate([y_train, np.array(new_y)])
        print(f"   Date augmentate: {len(X_augmented)} samples (+{len(new_X)} noi)")
    else:
        X_augmented = X_train
        y_augmented = y_train
        print(f"   Fără augmentări aplicabile")
    
    return X_augmented, y_augmented


# =============================================================================
# FUNCȚII ANTRENARE
# =============================================================================

def load_data():
    """Încarcă datele."""
    data_dir = PROJECT_ROOT / "data"
    
    X_train = np.load(data_dir / "train" / "X_train.npy")
    y_train = np.load(data_dir / "train" / "y_train.npy")
    X_val = np.load(data_dir / "validation" / "X_val.npy")
    y_val = np.load(data_dir / "validation" / "y_val.npy")
    X_test = np.load(data_dir / "test" / "X_test.npy")
    y_test = np.load(data_dir / "test" / "y_test.npy")
    
    print(f"✅ Date încărcate: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Creează DataLoaders."""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Antrenare o epocă."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion, device):
    """Validare."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    
    return total_loss / len(val_loader), correct / total


def evaluate_test(model, X_test, y_test, device):
    """Evaluare pe test set."""
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predictions = outputs.max(1)
        predictions = predictions.cpu().numpy()
    
    accuracy = np.mean(predictions == y_test)
    f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
    
    return accuracy, f1, predictions


# =============================================================================
# EXPORT ONNX - Nivel 3
# =============================================================================

def export_onnx(model, input_size, save_path):
    """
    Exportă modelul în format ONNX pentru deployment.
    """
    print(f"\n📦 Export ONNX...")
    
    model.eval()
    
    # Input dummy pentru trace
    dummy_input = torch.randn(1, input_size)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"   ✅ Model ONNX salvat: {save_path}")
    
    # Verificare dimensiune fișier
    file_size = os.path.getsize(save_path) / 1024  # KB
    print(f"   📊 Dimensiune: {file_size:.2f} KB")
    
    return save_path


def benchmark_onnx(onnx_path, input_size, num_iterations=100):
    """
    Benchmark latență pentru modelul ONNX.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("   ⚠️ onnxruntime nu este instalat. Instalare...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime", "-q"])
        import onnxruntime as ort
    
    print(f"\n⏱️ Benchmark latență ONNX ({num_iterations} iterații)...")
    
    # Creăm sesiune ONNX
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    
    # Input de test
    test_input = np.random.randn(1, input_size).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        session.run(None, {input_name: test_input})
    
    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        session.run(None, {input_name: test_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"   Latență medie: {avg_latency:.3f} ms")
    print(f"   Latență min:   {min_latency:.3f} ms")
    print(f"   Latență max:   {max_latency:.3f} ms")
    print(f"   Latență P95:   {p95_latency:.3f} ms")
    
    if avg_latency < 50:
        print(f"   ✅ PASS: Latență < 50ms (cerință Nivel 3)")
    else:
        print(f"   ⚠️ Latență > 50ms")
    
    return {
        'avg_ms': avg_latency,
        'min_ms': min_latency,
        'max_ms': max_latency,
        'p95_ms': p95_latency
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("🧠 Antrenare FINALĂ (Nivel 2 + Nivel 3 Complet)")
    print("=" * 60)
    
    # Config complet
    config = {
        'hidden_layers': [128, 64],
        'dropout': 0.3,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'epochs': 150,
        'patience': 10,
        'min_delta': 0.001,
        'lr_scheduler_factor': 0.5,
        'lr_scheduler_patience': 5,
        'augmentation_factor': 0.3,
    }
    
    print(f"\n📋 Configurație completă:")
    print(f"   Hidden layers: {config['hidden_layers']}")
    print(f"   Dropout: {config['dropout']}")
    print(f"   Weight decay: {config['weight_decay']}")
    print(f"   LR Scheduler: ReduceLROnPlateau (factor={config['lr_scheduler_factor']}, patience={config['lr_scheduler_patience']})")
    print(f"   Augmentare: {config['augmentation_factor']*100}% date noi")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Load preprocessing params
    with open(PROJECT_ROOT / "config" / "preprocessing_params.pkl", 'rb') as f:
        params = pickle.load(f)
    
    vocab = params['vocab']
    
    # AUGMENTARE DATE - Nivel 2
    X_train_aug, y_train_aug = augment_dataset(
        X_train, y_train, vocab, 
        augmentation_factor=config['augmentation_factor']
    )
    
    input_size = X_train_aug.shape[1]
    output_size = params['num_classes']
    
    print(f"\n📊 Model: {input_size} → {config['hidden_layers']} → {output_size}")
    
    # Create model
    model = SimpleNeuralNetwork(
        input_size=input_size,
        hidden_layers=config['hidden_layers'],
        output_size=output_size,
        dropout=config['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parametri: {total_params:,}")
    
    # Loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # LEARNING RATE SCHEDULER - Nivel 2
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['lr_scheduler_factor'],
        patience=config['lr_scheduler_patience']
    )
    
    # DataLoaders cu date augmentate
    train_loader, val_loader = create_dataloaders(
        X_train_aug, y_train_aug, X_val, y_val, config['batch_size']
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print("🚀 Începere antrenare cu LR Scheduler + Augmentări...")
    print(f"{'='*60}")
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # LR Scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(new_lr)
        
        gap = (train_acc - val_acc) * 100
        gap_symbol = "✓" if abs(gap) < 10 else "⚠️"
        
        if (epoch + 1) % 5 == 0 or epoch < 5:
            lr_change = " 📉LR" if new_lr < old_lr else ""
            print(f"Epoca {epoch+1:3d} | Train: {train_loss:.4f} ({train_acc*100:.1f}%) | "
                  f"Val: {val_loss:.4f} ({val_acc*100:.1f}%) | Gap: {gap:.1f}% {gap_symbol}{lr_change}")
        
        # Early stopping
        if val_loss < best_val_loss - config['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\n⏹️ Early stopping la epoca {epoch+1} (patience={config['patience']})")
                break
    
    # Restaurare best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Restaurat modelul cu best val_loss={best_val_loss:.4f}")
    
    # Evaluare test
    print(f"\n{'='*60}")
    print("📈 Evaluare pe test set...")
    print(f"{'='*60}")
    
    test_acc, test_f1, predictions = evaluate_test(model, X_test, y_test, device)
    
    # Calculăm accuracy pe train și val cu best model
    train_acc_final = history['train_acc'][-1] if history['train_acc'] else 0
    val_acc_final = history['val_acc'][-1] if history['val_acc'] else 0
    
    print(f"\n📊 REZULTATE FINALE:")
    print(f"   Train Accuracy:  {train_acc_final*100:.2f}%")
    print(f"   Val Accuracy:    {val_acc_final*100:.2f}%")
    print(f"   Test Accuracy:   {test_acc*100:.2f}%")
    print(f"   Test F1 (macro): {test_f1:.4f}")
    print(f"   Gap Train-Val:   {(train_acc_final-val_acc_final)*100:.2f}%")
    
    # Verificare obiective
    print(f"\n🎯 Verificare obiective:")
    print(f"   Accuracy ≥ 65%: {'✅ DA' if test_acc >= 0.65 else '❌ NU'} ({test_acc*100:.2f}%)")
    print(f"   F1 ≥ 0.60:      {'✅ DA' if test_f1 >= 0.60 else '❌ NU'} ({test_f1:.4f})")
    print(f"   Accuracy ≥ 75%: {'✅ DA' if test_acc >= 0.75 else '❌ NU'} (Nivel 2)")
    print(f"   F1 ≥ 0.70:      {'✅ DA' if test_f1 >= 0.70 else '❌ NU'} (Nivel 2)")
    
    # Salvare model PyTorch
    model_path = PROJECT_ROOT / "models" / "trained_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'input_size': input_size,
        'output_size': output_size,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'epochs_trained': len(history['train_loss']),
        'features': ['augmentation', 'lr_scheduler', 'early_stopping', 'weight_decay']
    }, model_path)
    print(f"\n💾 Model salvat: {model_path}")
    
    # EXPORT ONNX - Nivel 3
    onnx_path = PROJECT_ROOT / "models" / "trained_model.onnx"
    export_onnx(model.cpu(), input_size, onnx_path)
    
    # BENCHMARK ONNX - Nivel 3
    benchmark_results = benchmark_onnx(onnx_path, input_size)
    
    # Salvare metrici
    metrics = {
        'test_accuracy': float(test_acc),
        'test_f1_macro': float(test_f1),
        'train_accuracy': float(train_acc_final),
        'val_accuracy': float(val_acc_final),
        'epochs_trained': len(history['train_loss']),
        'best_val_loss': float(best_val_loss),
        'features_used': {
            'early_stopping': True,
            'lr_scheduler': 'ReduceLROnPlateau',
            'augmentation': f"{config['augmentation_factor']*100}%",
            'weight_decay': config['weight_decay']
        },
        'onnx_benchmark': benchmark_results
    }
    
    metrics_path = PROJECT_ROOT / "results" / "test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrici salvate: {metrics_path}")
    
    # Salvare training history cu LR
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
        'learning_rate': history['lr']
    })
    history_path = PROJECT_ROOT / "results" / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    
    # Grafic cu LR
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss în timpul antrenării')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot([x*100 for x in history['train_acc']], label='Train Acc', color='blue')
    axes[1].plot([x*100 for x in history['val_acc']], label='Val Acc', color='orange')
    axes[1].axhline(y=65, color='red', linestyle='--', label='Target 65%')
    axes[1].axhline(y=75, color='green', linestyle='--', label='Target 75% (N2)')
    axes[1].set_xlabel('Epoca')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy în timpul antrenării')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(history['lr'], label='Learning Rate', color='purple')
    axes[2].set_xlabel('Epoca')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plot_path = PROJECT_ROOT / "results" / "training_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📊 Grafic salvat: {plot_path}")
    
    print(f"\n{'='*60}")
    print("✅ Antrenare FINALĂ completă!")
    print(f"   📦 ONNX: {onnx_path}")
    print(f"   ⏱️ Latență: {benchmark_results['avg_ms']:.2f}ms (cerință: <50ms)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
