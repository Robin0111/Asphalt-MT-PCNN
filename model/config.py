"""
Configuration File for MT-PCNN

This file contains default configurations and can be imported
to customize model behavior without modifying source code.
"""

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
DATA_CONFIG = {
    'file_path': 'Ex_data.xlsx',
    'test_size': 0.2,
    'random_state': 42,
}

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
MODEL_CONFIG = {
    'hidden_layers': 3,
    'neurons_per_layer': 64,
    'activation': 'tanh',  # Options: 'tanh', 'relu', 'swish'
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
TRAINING_CONFIG = {
    'epochs': 3000,
    'learning_rate': 0.001,
    'physics_weight': 0.1,
    'batch_size': 32,
    'patience': 200,  # Early stopping patience
    'n_restarts': 5,  # Number of random restarts
    'val_split': 0.2,  # Validation split for random restarts
}

# =============================================================================
# CROSS-VALIDATION CONFIGURATION
# =============================================================================
CV_CONFIG = {
    'n_folds': 5,
    'epochs': 2000,
    'shuffle': True,
    'random_state': 42,
}

# =============================================================================
# BAYESIAN OPTIMIZATION CONFIGURATION
# =============================================================================
OPTIMIZATION_CONFIG = {
    'n_calls': 15,  # Number of optimization iterations
    'random_state': 42,
    'acq_func': 'EI',  # Acquisition function: 'EI', 'PI', 'LCB'
    
    # Search space
    'search_space': {
        'hidden_layers': (2, 5),
        'neurons_per_layer': (16, 512),
        'learning_rate': (0.0001, 0.01),  # Log-uniform
        'physics_weight': (0.1, 1.0),
        'activation': ['tanh', 'relu', 'swish'],
    },
    
    # Training settings for optimization
    'opt_epochs': 500,
    'opt_patience': 50,
}

# =============================================================================
# SENSITIVITY ANALYSIS CONFIGURATION
# =============================================================================
SOBOL_CONFIG = {
    'n_samples': 1024,
    'calc_second_order': True,
}

# =============================================================================
# C-VALUE ESTIMATION CONFIGURATION
# =============================================================================
C_VALUE_CONFIG = {
    'c_min': 0.5,   # Minimum allowed c value
    'c_max': 1.5,   # Maximum allowed c value
    'default_c': 1.0,  # Default if estimation fails
    'min_derivative': 1e-6,  # Minimum derivative for valid estimation
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
OUTPUT_CONFIG = {
    'model_save_dir': 'saved_models',
    'model_name': 'mt_pcnn_model',
    'results_filename': 'MT_PCNN_Results.xlsx',
}

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE_CONFIG = {
    'use_cuda': True,  # Set to False to force CPU
}


def get_config(config_name):

    configs = {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'cv': CV_CONFIG,
        'optimization': OPTIMIZATION_CONFIG,
        'sobol': SOBOL_CONFIG,
        'c_value': C_VALUE_CONFIG,
        'output': OUTPUT_CONFIG,
        'device': DEVICE_CONFIG,
    }
    
    return configs.get(config_name, {})


# Example usage in main.py:
# from config import MODEL_CONFIG, TRAINING_CONFIG
# model_params = MODEL_CONFIG
# trainer.train_model(**TRAINING_CONFIG)
