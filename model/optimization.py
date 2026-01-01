"""
Hyperparameter Optimization Module

This module implements Bayesian optimization using Gaussian Processes
for automated hyperparameter tuning of the MT-PCNN model.
"""

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


def bayesian_optimization(train_function, n_calls=20, random_state=42):

    # Define hyperparameter search space
    space = [
        Integer(2, 5, name='hidden_layers'),
        Integer(16, 512, name='neurons_per_layer'),
        Real(0.0001, 0.01, name='learning_rate', prior='log-uniform'),
        Real(0.1, 1.0, name='physics_weight'),
        Categorical(['tanh', 'relu', 'swish'], name='activation')
    ]
    
    @use_named_args(space)
    def objective(**params):
        """Wrapper function for optimization."""
        return train_function(**params)
    
    print("\nStarting Bayesian optimization...")
    print("=" * 60)
    print(f"Search space:")
    print(f"  hidden_layers: [2, 5]")
    print(f"  neurons_per_layer: [16, 512]")
    print(f"  learning_rate: [0.0001, 0.01] (log-uniform)")
    print(f"  physics_weight: [0.1, 1.0]")
    print(f"  activation: ['tanh', 'relu', 'swish']")
    print(f"\nTotal optimization calls: {n_calls}")
    print("=" * 60)
    
    # Run optimization
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        n_initial_points=max(5, n_calls // 4),
        random_state=random_state,
        verbose=True,
        acq_func='EI',  # Expected Improvement
        n_jobs=1
    )
    
    # Extract best parameters
    best_params = {
        'hidden_layers': result.x[0],
        'neurons_per_layer': result.x[1],
        'learning_rate': result.x[2],
        'physics_weight': result.x[3],
        'activation': result.x[4]
    }
    
    print("\n" + "=" * 60)
    print("Bayesian optimization completed!")
    print("=" * 60)
    print(f"Best hyperparameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest score (to minimize): {result.fun:.6f}")
    print("=" * 60)
    
    return best_params, result


class OptimizationObjective:
    
    def __init__(self, trainer, X_train, y_train, train_mixture_ids,
                 c_values_dict, scaler_X, scaler_log_modulus, scaler_phase_angle):
        self.trainer = trainer
        self.X_train = X_train
        self.y_train = y_train
        self.train_mixture_ids = train_mixture_ids
        self.c_values_dict = c_values_dict
        self.scaler_X = scaler_X
        self.scaler_log_modulus = scaler_log_modulus
        self.scaler_phase_angle = scaler_phase_angle
        self.optimization_history = []
    
    def __call__(self, hidden_layers, neurons_per_layer, learning_rate,
                 physics_weight, activation):

        try:
            params = {
                'hidden_layers': hidden_layers,
                'neurons_per_layer': neurons_per_layer,
                'learning_rate': learning_rate,
                'physics_weight': physics_weight,
                'activation': activation
            }
            
            print(f"\nTesting: {params}")
            
            # Update trainer parameters
            self.trainer.model_params = params
            
            # Train model with reduced epochs for optimization speed
            model, val_loss, _ = self.trainer.train_single_model(
                self.X_train, self.y_train,
                self.X_train[:100],  # Small validation set
                self.y_train[:100],
                self.train_mixture_ids,
                self.c_values_dict,
                epochs=500,  # Shorter for optimization
                learning_rate=learning_rate,
                physics_weight=physics_weight,
                batch_size=32,
                patience=50,
                verbose=False
            )
            
            # Calculate score (validation loss)
            score = val_loss
            
            # Check for invalid results
            if score is None or np.isnan(score) or np.isinf(score):
                score = 1000.0
            
            # Store in history
            self.optimization_history.append({
                'hidden_layers': hidden_layers,
                'neurons_per_layer': neurons_per_layer,
                'learning_rate': learning_rate,
                'physics_weight': physics_weight,
                'activation': activation,
                'score': score
            })
            
            print(f"Score: {score:.6f}")
            print("-" * 60)
            
            return score
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1000.0


def format_optimization_history(history):

    import pandas as pd
    
    if not history:
        return pd.DataFrame()
    
    df = pd.DataFrame(history)
    
    # Sort by score (best first)
    df = df.sort_values('score')
    
    # Add rank column
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    return df
