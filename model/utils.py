"""
Utilities Module

This module provides utility functions for saving/loading models,
saving results to Excel, and other helper functions.
"""

import os
import torch
import joblib
import pandas as pd
from datetime import datetime


def save_model(model, model_params, scaler_X, scaler_log_modulus, 
               scaler_phase_angle, c_values_dict, feature_names,
               save_dir='saved_models', model_name='mt_pcnn_model'):

    # Create directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(save_dir, f"{model_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': model_params,
        'c_values_dict': c_values_dict,
        'feature_names': feature_names
    }, model_path)
    
    # Save scalers
    scaler_path = os.path.join(model_dir, 'scalers.pkl')
    joblib.dump({
        'scaler_X': scaler_X,
        'scaler_log_modulus': scaler_log_modulus,
        'scaler_phase_angle': scaler_phase_angle
    }, scaler_path)
    
    print(f"\nModel saved successfully to: {model_dir}")
    print(f"  - Model weights: {model_path}")
    print(f"  - Scalers: {scaler_path}")
    
    return model_dir


def load_model(model_dir, device='cpu'):

    from models import PCNNModel
    
    device = torch.device(device)
    
    # Load model state
    model_path = os.path.join(model_dir, 'model.pth')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct model
    model_params = checkpoint['model_params']
    input_dim = len(checkpoint['feature_names'])
    
    model = PCNNModel(
        input_dim=input_dim,
        hidden_layers=model_params['hidden_layers'],
        neurons_per_layer=model_params['neurons_per_layer'],
        activation=model_params['activation']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scalers
    scaler_path = os.path.join(model_dir, 'scalers.pkl')
    scalers = joblib.load(scaler_path)
    
    print(f"\nModel loaded successfully from: {model_dir}")
    
    return {
        'model': model,
        'model_params': model_params,
        'scalers': scalers,
        'c_values_dict': checkpoint['c_values_dict'],
        'feature_names': checkpoint['feature_names']
    }


def save_results_to_excel(filename, test_results=None, cv_results=None,
                          sobol_results=None, c_values_dict=None,
                          model_params=None, optimization_history=None):

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Test predictions
        if test_results is not None and 'predictions' in test_results:
            pred_df = pd.DataFrame({
                'Predicted_Log_Modulus': test_results['predictions'][:, 0],
                'Predicted_Phase_Angle': test_results['predictions'][:, 1],
                'True_Log_Modulus': test_results.get('true_values', [None]*len(test_results['predictions']))[:, 0] if 'true_values' in test_results else None,
                'True_Phase_Angle': test_results.get('true_values', [None]*len(test_results['predictions']))[:, 1] if 'true_values' in test_results else None,
            })
            
            if 'true_values' in test_results:
                pred_df['Error_Log_Modulus'] = pred_df['Predicted_Log_Modulus'] - pred_df['True_Log_Modulus']
                pred_df['Error_Phase_Angle'] = pred_df['Predicted_Phase_Angle'] - pred_df['True_Phase_Angle']
            
            pred_df.to_excel(writer, sheet_name='Test_Predictions', index=False)
        
        # Test metrics
        if test_results is not None:
            metrics_data = {
                'Metric': ['R²', 'RMSE', 'MAE'],
                'Log_Modulus': [
                    test_results.get('r2_log'),
                    test_results.get('rmse_log'),
                    test_results.get('mae_log')
                ],
                'Phase_Angle': [
                    test_results.get('r2_phase'),
                    test_results.get('rmse_phase'),
                    test_results.get('mae_phase')
                ]
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name='Test_Metrics', index=False)
        
        # Cross-validation results
        if cv_results is not None and len(cv_results) > 0:
            cv_data = []
            for result in cv_results:
                cv_data.append({
                    'Fold': result.get('fold'),
                    'R2_Log_Modulus': result.get('r2_log'),
                    'R2_Phase_Angle': result.get('r2_phase'),
                    'RMSE_Log_Modulus': result.get('rmse_log'),
                    'RMSE_Phase_Angle': result.get('rmse_phase'),
                    'MAE_Log_Modulus': result.get('mae_log'),
                    'MAE_Phase_Angle': result.get('mae_phase')
                })
            
            cv_df = pd.DataFrame(cv_data)
            
            # Add summary statistics
            summary_row = {
                'Fold': 'Mean ± Std',
                'R2_Log_Modulus': f"{cv_df['R2_Log_Modulus'].mean():.4f} ± {cv_df['R2_Log_Modulus'].std():.4f}",
                'R2_Phase_Angle': f"{cv_df['R2_Phase_Angle'].mean():.4f} ± {cv_df['R2_Phase_Angle'].std():.4f}",
                'RMSE_Log_Modulus': f"{cv_df['RMSE_Log_Modulus'].mean():.4f} ± {cv_df['RMSE_Log_Modulus'].std():.4f}",
                'RMSE_Phase_Angle': f"{cv_df['RMSE_Phase_Angle'].mean():.4f} ± {cv_df['RMSE_Phase_Angle'].std():.4f}",
                'MAE_Log_Modulus': f"{cv_df['MAE_Log_Modulus'].mean():.4f} ± {cv_df['MAE_Log_Modulus'].std():.4f}",
                'MAE_Phase_Angle': f"{cv_df['MAE_Phase_Angle'].mean():.4f} ± {cv_df['MAE_Phase_Angle'].std():.4f}"
            }
            cv_df = pd.concat([cv_df, pd.DataFrame([summary_row])], ignore_index=True)
            
            cv_df.to_excel(writer, sheet_name='Cross_Validation', index=False)
        
        # C values
        if c_values_dict is not None:
            c_df = pd.DataFrame([
                {'Mixture_ID': k, 'C_Value': v}
                for k, v in c_values_dict.items()
            ])
            c_df.to_excel(writer, sheet_name='C_Values', index=False)
        
        # Sobol indices
        if sobol_results is not None:
            from sensitivity_analysis import extract_sobol_dataframes
            sobol_dfs = extract_sobol_dataframes(sobol_results)
            
            if sobol_dfs['first_order_log'] is not None:
                sobol_dfs['first_order_log'].to_excel(
                    writer, sheet_name='Sobol_Log_Modulus', index=False
                )
            
            if sobol_dfs['first_order_phase'] is not None:
                sobol_dfs['first_order_phase'].to_excel(
                    writer, sheet_name='Sobol_Phase_Angle', index=False
                )
            
            if sobol_dfs['second_order_log'] is not None:
                sobol_dfs['second_order_log'].to_excel(
                    writer, sheet_name='Sobol_Second_Order_Log', index=False
                )
            
            if sobol_dfs['second_order_phase'] is not None:
                sobol_dfs['second_order_phase'].to_excel(
                    writer, sheet_name='Sobol_Second_Order_Phase', index=False
                )
        
        # Model hyperparameters
        if model_params is not None:
            params_df = pd.DataFrame([
                {'Parameter': k, 'Value': v}
                for k, v in model_params.items()
            ])
            params_df.to_excel(writer, sheet_name='Model_Parameters', index=False)
        
        # Optimization history
        if optimization_history is not None and len(optimization_history) > 0:
            from optimization import format_optimization_history
            opt_df = format_optimization_history(optimization_history)
            opt_df.to_excel(writer, sheet_name='Optimization_History', index=False)
    
    print(f"\nResults saved to: {filename}")
    return filename
