"""
Modul Neural Network
====================

Implementare rețea neuronală cu PyTorch.
"""

from .model import (
    NeuralNetwork,
    NeuralNetworkTrainer,
    create_model_from_config
)

__all__ = [
    'NeuralNetwork',
    'NeuralNetworkTrainer',
    'create_model_from_config'
]
