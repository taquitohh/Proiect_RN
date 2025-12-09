# 🧠 Neural Network Module

## Descriere
Acest modul conține **arhitectura rețelei neuronale** pentru clasificarea intențiilor din comenzi text.

## Structura
```
neural_network/
├── __init__.py
├── model.py           # Definiție model și trainer
└── README.md          # Acest fișier
```

## Arhitectura Modelului

### Clasa `NeuralNetwork`
Rețea neuronală feed-forward cu arhitectură configurabilă:

```python
NeuralNetwork(
    input_size=768,        # Dimensiune embedding text
    hidden_layers=[256, 128, 64],  # Straturi ascunse
    output_size=20,        # Număr de intenții
    activation='relu',     # Funcție activare
    dropout=0.2            # Regularizare
)
```

### Justificarea arhitecturii
1. **Input 768 dimensiuni** - Compatibil cu embeddings de la modele pre-antrenate (BERT, etc.)
2. **3 straturi ascunse** - Suficient pentru clasificare text de complexitate medie
3. **Reducere progresivă (256→128→64)** - Extragere features ierarhică
4. **ReLU activation** - Standard pentru rețele feed-forward, evită vanishing gradients
5. **Dropout 20%** - Previne overfitting pe dataset mic

### Funcții principale
- `forward(x)` - Forward pass prin rețea
- `predict(x)` - Predicție cu softmax pentru clasificare
- `save_model(path)` - Salvare weights
- `load_model(path)` - Încărcare weights

### Clasa `NeuralNetworkTrainer`
Gestionează antrenarea modelului:
- Loss function: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Early stopping configurabil

## Status actual (Etapa 4)
- ✅ Arhitectură definită și compilată
- ✅ Model poate fi salvat/încărcat
- ⏳ Antrenare preliminară (weights inițializați random)
- ⏳ Optimizare hiperparametri (Etapa 5)

## Comenzi de test
```bash
# Verificare că modelul se poate instanția
python -c "from neural_network.model import NeuralNetwork; m = NeuralNetwork(100, [64, 32], 10); print(m)"

# Test forward pass
python -c "
import torch
from neural_network.model import NeuralNetwork
model = NeuralNetwork(100, [64, 32], 10)
x = torch.randn(1, 100)
print('Output shape:', model(x).shape)
"
```

## Configurare (config/model_config.yaml)
```yaml
model:
  input_size: 768
  hidden_layers: [256, 128, 64]
  output_size: 20
  activation: relu
  dropout: 0.2
  
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 10
```
