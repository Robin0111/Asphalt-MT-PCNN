"""
Training Module

This module contains the training logic for MT-PCNN including:
- Model training with physics constraints
- Random restart strategy
- Cross-validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from models import PCNNModel
from physics_constraints import kramers_kronig_loss


class Trainer:
    
    def __init__(self, model_params, device='cpu'):
        self.model_params = model_params
        self.device = torch.device(device)
        self.model = None
        self.best_model = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'data_loss': [],
            'physics_loss': []
        }
    
    def build_model(self, input_dim):

        model = PCNNModel(
            input_dim=input_dim,
            hidden_layers=self.model_params['hidden_layers'],
            neurons_per_layer=self.model_params['neurons_per_layer'],
            activation=self.model_params['activation']
        ).to(self.device)
        
        return model
    
    def train_epoch(self, model, train_loader, optimizer, physics_weight, c_values_dict):

        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        n_batches = 0
        
        for batch_X, batch_y, batch_mixture_ids in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Get c values for this batch
            c_values = np.array([c_values_dict[mid] for mid in batch_mixture_ids])
            c_values_tensor = torch.tensor(c_values, dtype=torch.float32).to(self.device)
            
            # Forward pass
            predictions = model(batch_X)
            
            # Data loss (MSE)
            data_loss = nn.MSELoss()(predictions, batch_y)
            
            # Physics constraint loss (Kramers-Kronig)
            physics_loss = kramers_kronig_loss(predictions, batch_X, c_values_tensor)
            
            # Combined loss
            loss = data_loss + physics_weight * physics_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item()
            n_batches += 1
        
        return (total_loss / n_batches, 
                total_data_loss / n_batches,
                total_physics_loss / n_batches)
    
    def validate(self, model, val_loader):

        model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y, _ in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = model(batch_X)
                loss = nn.MSELoss()(predictions, batch_y)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def train_single_model(self, X_train, y_train, X_val, y_val, 
                          train_mixture_ids, c_values_dict,
                          epochs=1000, learning_rate=0.001, 
                          physics_weight=0.1, batch_size=32, 
                          patience=100, verbose=True):

        # Build model
        input_dim = X_train.shape[1]
        model = self.build_model(input_dim)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
            torch.LongTensor(train_mixture_ids)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
            torch.LongTensor(np.zeros(len(X_val)))  # Dummy for consistency
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'data_loss': [],
            'physics_loss': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, data_loss, physics_loss = self.train_epoch(
                model, train_loader, optimizer, physics_weight, c_values_dict
            )
            
            # Validate
            val_loss = self.validate(model, val_loader)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['data_loss'].append(data_loss)
            history['physics_loss'].append(physics_loss)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"Data: {data_loss:.6f}, "
                      f"Physics: {physics_loss:.6f}")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return model, best_val_loss, history
    
    def train_with_random_restarts(self, X_train, y_train, train_mixture_ids, 
                                   c_values_dict, n_restarts=5, 
                                   val_split=0.2, **train_kwargs):

        print(f"\nTraining with {n_restarts} random restarts...")
        print("=" * 60)
        
        # Split training data for validation
        n_val = int(len(X_train) * val_split)
        indices = np.random.permutation(len(X_train))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train_split = X_train[train_indices]
        y_train_split = y_train[train_indices]
        X_val_split = X_train[val_indices]
        y_val_split = y_train[val_indices]
        train_mixture_ids_split = train_mixture_ids[train_indices]
        
        best_overall_model = None
        best_overall_loss = float('inf')
        
        for restart in range(n_restarts):
            print(f"\nRestart {restart + 1}/{n_restarts}")
            print("-" * 60)
            
            model, val_loss, _ = self.train_single_model(
                X_train_split, y_train_split,
                X_val_split, y_val_split,
                train_mixture_ids_split,
                c_values_dict,
                **train_kwargs
            )
            
            print(f"Restart {restart + 1} final validation loss: {val_loss:.6f}")
            
            if val_loss < best_overall_loss:
                best_overall_loss = val_loss
                best_overall_model = model
                print(f"New best model found! Loss: {val_loss:.6f}")
        
        print("\n" + "=" * 60)
        print(f"Best model from {n_restarts} restarts - Loss: {best_overall_loss:.6f}")
        
        self.best_model = best_overall_model
        return best_overall_model, best_overall_loss
