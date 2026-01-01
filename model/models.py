"""
Neural Network Models for MT-PCNN

This module contains the neural network architecture definitions for the
Multi-Task Physics-Constrained Neural Network (MT-PCNN) for asphalt mixture
property prediction.
"""

import torch
import torch.nn as nn


class PCNNModel(nn.Module):

    
    def __init__(self, input_dim, hidden_layers, neurons_per_layer, activation='tanh'):
        super(PCNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, neurons_per_layer))
        layers.append(self._get_activation(activation))
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(self._get_activation(activation))
        
        # Output layer (2 outputs: log|E*| and phase angle)
        layers.append(nn.Linear(neurons_per_layer, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.init_weights()
    
    def _get_activation(self, activation):

        if activation == 'tanh':
            return nn.Tanh()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'swish':
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def init_weights(self):
        """
        Initialize network weights using Xavier (Glorot) normal initialization.
        Biases are initialized to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        return self.network(x)
