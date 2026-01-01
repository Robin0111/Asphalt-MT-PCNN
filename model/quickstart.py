"""
Quick Start Example for MT-PCNN

This script demonstrates basic usage of the MT-PCNN model
with minimal configuration.
"""

import torch
from data_loader import DataLoader
from physics_constraints import estimate_c_values
from models import PCNNModel
from trainer import Trainer
from evaluation import evaluate_model, print_evaluation_results
from utils import save_model, save_results_to_excel

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
print("\n" + "=" * 60)
print("Loading and preparing data...")
print("=" * 60)

loader = DataLoader()
data = loader.load_data('Ex_data.xlsx')

# Estimate c values
c_values_dict = estimate_c_values(
    data['mixture_ids'],
    data['log_omega'],
    data['log_modulus'],
    data['phase_angle']
)

# Prepare train/test split
split_data = loader.prepare_data(test_size=0.2, random_state=42)

# =============================================================================
# 2. TRAIN MODEL
# =============================================================================
print("\n" + "=" * 60)
print("Training model...")
print("=" * 60)

# Define model parameters
model_params = {
    'hidden_layers': 3,
    'neurons_per_layer': 64,
    'activation': 'tanh'
}

# Create trainer
trainer = Trainer(model_params, device=device)

# Train with random restarts
best_model, best_loss = trainer.train_with_random_restarts(
    X_train=split_data['X_train_scaled'],
    y_train=split_data['y_train_scaled'],
    train_mixture_ids=split_data['train_mixture_ids'],
    c_values_dict=c_values_dict,
    n_restarts=3,
    epochs=1000,
    learning_rate=0.001,
    physics_weight=0.1,
    batch_size=32,
    verbose=True
)

# =============================================================================
# 3. EVALUATE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("Evaluating model on test set...")
print("=" * 60)

test_results = evaluate_model(
    best_model,
    split_data['X_test_scaled'],
    split_data['y_test'],
    loader.scaler_log_modulus,
    loader.scaler_phase_angle,
    device=device
)

test_results['true_values'] = split_data['y_test']

print_evaluation_results(test_results, "Test Set Results")

# =============================================================================
# 4. SAVE RESULTS
# =============================================================================
print("\n" + "=" * 60)
print("Saving model and results...")
print("=" * 60)

# Save model
model_dir = save_model(
    best_model, model_params,
    loader.scaler_X,
    loader.scaler_log_modulus,
    loader.scaler_phase_angle,
    c_values_dict,
    loader.feature_names
)

# Save results to Excel
save_results_to_excel(
    'QuickStart_Results.xlsx',
    test_results=test_results,
    c_values_dict=c_values_dict,
    model_params=model_params
)

print("\n" + "=" * 60)
print("Quick start example completed!")
print("=" * 60)
print(f"Model saved to: {model_dir}")
print(f"Results saved to: QuickStart_Results.xlsx")
