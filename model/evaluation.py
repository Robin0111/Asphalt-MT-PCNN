"""
Evaluation and Metrics Module

This module provides functions for model evaluation, performance metrics,
and cross-validation.
"""

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold


def evaluate_model(model, X_test, y_test, scaler_log_modulus, scaler_phase_angle, device='cpu'):

    model.eval()
    device = torch.device(device)
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        predictions_scaled = model(X_test_tensor).cpu().numpy()
    
    # Inverse transform predictions
    pred_log_modulus = scaler_log_modulus.inverse_transform(
        predictions_scaled[:, 0].reshape(-1, 1)
    ).ravel()
    
    pred_phase_angle = scaler_phase_angle.inverse_transform(
        predictions_scaled[:, 1].reshape(-1, 1)
    ).ravel()
    
    predictions = np.column_stack([pred_log_modulus, pred_phase_angle])
    
    # Extract true values
    true_log_modulus = y_test[:, 0]
    true_phase_angle = y_test[:, 1]
    
    # Calculate metrics for log modulus
    r2_log = r2_score(true_log_modulus, pred_log_modulus)
    rmse_log = np.sqrt(mean_squared_error(true_log_modulus, pred_log_modulus))
    mae_log = mean_absolute_error(true_log_modulus, pred_log_modulus)
    
    # Calculate metrics for phase angle
    r2_phase = r2_score(true_phase_angle, pred_phase_angle)
    rmse_phase = np.sqrt(mean_squared_error(true_phase_angle, pred_phase_angle))
    mae_phase = mean_absolute_error(true_phase_angle, pred_phase_angle)
    
    return {
        'predictions': predictions,
        'r2_log': r2_log,
        'r2_phase': r2_phase,
        'rmse_log': rmse_log,
        'rmse_phase': rmse_phase,
        'mae_log': mae_log,
        'mae_phase': mae_phase
    }


def print_evaluation_results(results, title="Model Evaluation Results"):

    print(f"\n{title}")
    print("=" * 60)
    print(f"{'Metric':<30} {'Log Modulus':<15} {'Phase Angle':<15}")
    print("-" * 60)
    print(f"{'R² Score':<30} {results['r2_log']:>14.4f} {results['r2_phase']:>14.4f}")
    print(f"{'RMSE':<30} {results['rmse_log']:>14.4f} {results['rmse_phase']:>14.4f}")
    print(f"{'MAE':<30} {results['mae_log']:>14.4f} {results['mae_phase']:>14.4f}")
    print("=" * 60)


def cross_validate(model_class, model_params, X, y, mixture_ids, c_values_dict,
                   scaler_X, scaler_log_modulus, scaler_phase_angle,
                   n_folds=5, epochs=1000, learning_rate=0.001, 
                   physics_weight=0.1, batch_size=32, device='cpu'):

    from trainer import Trainer
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_results = []
    
    print(f"\nPerforming {n_folds}-fold cross-validation...")
    print("=" * 60)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"\nFold {fold}/{n_folds}")
        print("-" * 60)
        
        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        mixture_ids_train = mixture_ids[train_idx]
        
        # Scale data
        scaler_X_fold = type(scaler_X)()
        scaler_log_fold = type(scaler_log_modulus)()
        scaler_phase_fold = type(scaler_phase_angle)()
        
        X_train_scaled = scaler_X_fold.fit_transform(X_train_fold)
        X_val_scaled = scaler_X_fold.transform(X_val_fold)
        
        y_train_log_scaled = scaler_log_fold.fit_transform(
            y_train_fold[:, 0].reshape(-1, 1)
        )
        y_train_phase_scaled = scaler_phase_fold.fit_transform(
            y_train_fold[:, 1].reshape(-1, 1)
        )
        y_train_scaled = np.column_stack([
            y_train_log_scaled.ravel(),
            y_train_phase_scaled.ravel()
        ])
        
        # Train model
        trainer = Trainer(model_params, device=device)
        
        # Simple training for CV (no restarts to save time)
        model, _, _ = trainer.train_single_model(
            X_train_scaled, y_train_scaled,
            X_val_scaled, y_train_scaled[:len(X_val_scaled)],  # Dummy validation
            mixture_ids_train, c_values_dict,
            epochs=epochs,
            learning_rate=learning_rate,
            physics_weight=physics_weight,
            batch_size=batch_size,
            verbose=False
        )
        
        # Evaluate
        results = evaluate_model(
            model, X_val_scaled, y_val_fold,
            scaler_log_fold, scaler_phase_fold, device
        )
        
        results['fold'] = fold
        cv_results.append(results)
        
        print(f"Fold {fold} - R²: {results['r2_log']:.4f} (log), "
              f"{results['r2_phase']:.4f} (phase)")
        print(f"Fold {fold} - RMSE: {results['rmse_log']:.4f} (log), "
              f"{results['rmse_phase']:.4f} (phase)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Cross-Validation Summary")
    print("=" * 60)
    
    avg_r2_log = np.mean([r['r2_log'] for r in cv_results])
    avg_r2_phase = np.mean([r['r2_phase'] for r in cv_results])
    avg_rmse_log = np.mean([r['rmse_log'] for r in cv_results])
    avg_rmse_phase = np.mean([r['rmse_phase'] for r in cv_results])
    
    std_r2_log = np.std([r['r2_log'] for r in cv_results])
    std_r2_phase = np.std([r['r2_phase'] for r in cv_results])
    std_rmse_log = np.std([r['rmse_log'] for r in cv_results])
    std_rmse_phase = np.std([r['rmse_phase'] for r in cv_results])
    
    print(f"Average R² (log modulus): {avg_r2_log:.4f} ± {std_r2_log:.4f}")
    print(f"Average R² (phase angle): {avg_r2_phase:.4f} ± {std_r2_phase:.4f}")
    print(f"Average RMSE (log modulus): {avg_rmse_log:.4f} ± {std_rmse_log:.4f}")
    print(f"Average RMSE (phase angle): {avg_rmse_phase:.4f} ± {std_rmse_phase:.4f}")
    print("=" * 60)
    
    return cv_results
