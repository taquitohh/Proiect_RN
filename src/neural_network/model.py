"""
Modul pentru implementarea rețelei neuronale.
==============================================

Acest modul conține:
- Definiția arhitecturii rețelei neuronale
- Funcții de antrenare și evaluare
- Utilități pentru salvare/încărcare modele
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Any, List, Tuple
import yaml
from tqdm import tqdm


class NeuralNetwork(nn.Module):
    """Rețea neuronală flexibilă cu arhitectură configurabilă."""
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        activation: str = 'relu',
        output_activation: str = 'softmax',
        dropout: float = 0.2
    ):
        """
        Inițializare rețea neuronală.
        
        Args:
            input_size: Dimensiunea intrării
            hidden_layers: Lista cu dimensiunile straturilor ascunse
            output_size: Dimensiunea ieșirii
            activation: Funcția de activare ('relu', 'tanh', 'sigmoid')
            output_activation: Funcția de activare pentru ieșire
            dropout: Rata de dropout
        """
        super(NeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        
        # Selectare funcție de activare
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        self.activation = activations.get(activation, nn.ReLU())
        
        # Construire straturi
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Strat de ieșire
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Funcție de activare pentru ieșire (pentru inferență)
        self.output_activation = output_activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicție cu funcție de activare aplicată."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            if self.output_activation == 'softmax':
                return torch.softmax(output, dim=1)
            elif self.output_activation == 'sigmoid':
                return torch.sigmoid(output)
            return output


class NeuralNetworkTrainer:
    """Trainer pentru rețeaua neuronală."""
    
    def __init__(
        self,
        model: NeuralNetwork,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Inițializare trainer.
        
        Args:
            model: Modelul de antrenat
            config: Configurația de antrenare
        """
        self.model = model
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self._setup_training()
    
    def _default_config(self) -> Dict[str, Any]:
        """Configurație implicită."""
        return {
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
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
    
    def _setup_training(self):
        """Configurare componente de antrenare."""
        config = self.config['training']
        
        # Optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9
            )
        else:
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=config['learning_rate']
            )
        
        # Loss function
        problem_type = self.config.get('model', {}).get('problem_type', 'classification')
        if problem_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
    
    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Creează DataLoader din numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y) if self.config.get('model', {}).get('problem_type', 'classification') == 'classification' else torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, List[float]]:
        """
        Antrenează modelul.
        
        Args:
            X_train: Date de antrenare
            y_train: Etichete de antrenare
            X_val: Date de validare
            y_val: Etichete de validare
            
        Returns:
            Dicționar cu istoricul antrenării
        """
        config = self.config['training']
        
        train_loader = self._create_dataloader(
            X_train, y_train,
            batch_size=config['batch_size']
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(
                X_val, y_val,
                batch_size=config['batch_size'],
                shuffle=False
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
                if self.config.get('model', {}).get('problem_type', 'classification') == 'classification':
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += y_batch.size(0)
                    train_correct += (predicted == y_batch).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            if val_loader:
                val_loss, val_acc = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Early stopping
                if config['early_stopping']['enabled']:
                    if val_loss < best_val_loss - config['early_stopping']['min_delta']:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= config['early_stopping']['patience']:
                            print(f"\nEarly stopping la epoca {epoch + 1}")
                            break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoca {epoch + 1}/{config['epochs']} - "
                          f"Loss: {avg_train_loss:.4f} - Acc: {train_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoca {epoch + 1}/{config['epochs']} - "
                          f"Loss: {avg_train_loss:.4f} - Acc: {train_acc:.4f}")
        
        return self.history
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validare model."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                val_loss += loss.item()
                
                if self.config.get('model', {}).get('problem_type', 'classification') == 'classification':
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        return avg_val_loss, val_acc
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluează modelul pe setul de test.
        
        Args:
            X_test: Date de test
            y_test: Etichete de test
            
        Returns:
            Dicționar cu metrici de evaluare
        """
        test_loader = self._create_dataloader(
            X_test, y_test,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        test_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                test_loss += loss.item()
                
                if self.config.get('model', {}).get('problem_type', 'classification') == 'classification':
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
        
        avg_test_loss = test_loss / len(test_loader)
        
        if self.config.get('model', {}).get('problem_type', 'classification') == 'classification':
            accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
            return {
                'test_loss': avg_test_loss,
                'accuracy': accuracy,
                'predictions': all_predictions,
                'labels': all_labels
            }
        
        return {'test_loss': avg_test_loss}
    
    def save_model(self, path: str):
        """Salvează modelul."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)
        print(f"Model salvat în: {path}")
    
    def load_model(self, path: str):
        """Încarcă modelul."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model încărcat din: {path}")


def create_model_from_config(config_path: str, input_size: int, output_size: int) -> NeuralNetwork:
    """
    Creează model din fișier de configurare.
    
    Args:
        config_path: Calea către fișierul de configurare
        input_size: Dimensiunea intrării
        output_size: Dimensiunea ieșirii
        
    Returns:
        Model NeuralNetwork configurat
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    arch = config['model']['architecture']
    
    return NeuralNetwork(
        input_size=input_size,
        hidden_layers=arch['hidden_layers'],
        output_size=output_size,
        activation=arch['activation'],
        output_activation=arch['output_activation'],
        dropout=arch['dropout']
    )


if __name__ == "__main__":
    # Exemplu de utilizare
    print("Testare rețea neuronală...")
    
    # Date sintetice
    np.random.seed(42)
    X_train = np.random.randn(800, 10).astype(np.float32)
    y_train = np.random.randint(0, 3, 800)
    X_val = np.random.randn(100, 10).astype(np.float32)
    y_val = np.random.randint(0, 3, 100)
    X_test = np.random.randn(100, 10).astype(np.float32)
    y_test = np.random.randint(0, 3, 100)
    
    # Creare model
    model = NeuralNetwork(
        input_size=10,
        hidden_layers=[64, 32],
        output_size=3
    )
    
    # Antrenare
    trainer = NeuralNetworkTrainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluare
    results = trainer.evaluate(X_test, y_test)
    print(f"\nAcuratețe test: {results['accuracy']:.4f}")
