"""
Main Script for MT-PCNN Training and Analysis

This script orchestrates the complete workflow for:
1. Data loading and preprocessing
2. C value estimation
3. Model training with random restarts
4. Cross-validation
5. Hyperparameter optimization
6. Sensitivity analysis
7. Results saving
"""

import torch
import numpy as np
import warnings

from data_loader import DataLoader
from physics_constraints import estimate_c_values
from models import PCNNModel
from trainer import Trainer
from evaluation import evaluate_model, print_evaluation_results, cross_validate
from sensitivity_analysis import sobol_sensitivity_analysis
from optimization import bayesian_optimization, OptimizationObjective
from utils import save_model, save_results_to_excel

warnings.filterwarnings('ignore')


def main():
    """
    Main function to run the complete MT-PCNN pipeline.
    """
    print("=" * 80)
    print("MT-PCNN: Multi-Task Physics-Constrained Neural Network")
    print("For Asphalt Mixture Dynamic Modulus and Phase Angle Prediction")
    print("With Implicit C Values Estimated from Data")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # =========================================================================
    # 1. DATA LOADING AND PREPARATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("=" * 80)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load data from Excel
    data = data_loader.load_data('Ex_data.xlsx', sheet_name='Sheet1')
    
    # Estimate c values from data
    c_values_dict = estimate_c_values(
        data['mixture_ids'],
        data['log_omega'],
        data['log_modulus'],
        data['phase_angle']
    )
    
    # Prepare train/test split
    split_data = data_loader.prepare_data(test_size=0.2, random_state=42)
    
    X_train_scaled = split_data['X_train_scaled']
    X_test_scaled = split_data['X_test_scaled']
    y_train_scaled = split_data['y_train_scaled']
    y_test_scaled = split_data['y_test_scaled']
    y_test = split_data['y_test']
    train_mixture_ids = split_data['train_mixture_ids']
    
    # =========================================================================
    # 2. HYPERPARAMETER OPTIMIZATION (OPTIONAL)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Default parameters
    model_params = {
        'hidden_layers': 3,
        'neurons_per_layer': 64,
        'activation': 'tanh'
    }
    
    run_optimization = input("\nRun Bayesian optimization for hyperparameter tuning? (y/n): ").lower() == 'y'
    
    optimization_history = []
    
    if run_optimization:
        n_calls = int(input("Number of optimization calls (default 15): ") or "15")
        
        # Create trainer for optimization
        trainer = Trainer(model_params, device=device)
        
        # Create optimization objective
        obj = OptimizationObjective(
            trainer, X_train_scaled, y_train_scaled, train_mixture_ids,
            c_values_dict, data_loader.scaler_X,
            data_loader.scaler_log_modulus, data_loader.scaler_phase_angle
        )
        
        # Run optimization
        best_params, opt_result = bayesian_optimization(obj, n_calls=n_calls)
        
        model_params.update(best_params)
        optimization_history = obj.optimization_history
    else:
        print("\nUsing default hyperparameters...")
        model_params['learning_rate'] = 0.001
        model_params['physics_weight'] = 0.1
    
    # =========================================================================
    # 3. MODEL TRAINING WITH RANDOM RESTARTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: MODEL TRAINING WITH RANDOM RESTARTS")
    print("=" * 80)
    
    # Get training parameters
    n_restarts = int(input("Number of random restarts for final training (default 5): ") or "5")
    
    learning_rate = model_params.get('learning_rate', 0.001)
    physics_weight = model_params.get('physics_weight', 0.1)
    
    # Create trainer
    trainer = Trainer(model_params, device=device)
    
    # Train with random restarts
    best_model, best_loss = trainer.train_with_random_restarts(
        X_train_scaled, y_train_scaled, train_mixture_ids,
        c_values_dict,
        n_restarts=n_restarts,
        val_split=0.2,
        epochs=3000,
        learning_rate=learning_rate,
        physics_weight=physics_weight,
        batch_size=32,
        patience=200,
        verbose=True
    )
    
    # =========================================================================
    # 4. CROSS-VALIDATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: CROSS-VALIDATION")
    print("=" * 80)
    
    run_cv = input("\nPerform 5-fold cross-validation? (y/n): ").lower() == 'y'
    
    cv_results = []
    if run_cv:
        cv_results = cross_validate(
            PCNNModel, model_params,
            split_data['X_train'], split_data['y_train'],
            train_mixture_ids, c_values_dict,
            data_loader.scaler_X,
            data_loader.scaler_log_modulus,
            data_loader.scaler_phase_angle,
            n_folds=5,
            epochs=2000,
            learning_rate=learning_rate,
            physics_weight=physics_weight,
            device=device
        )
    
    # =========================================================================
    # 5. TEST SET EVALUATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: TEST SET EVALUATION")
    print("=" * 80)
    
    test_results = evaluate_model(
        best_model, X_test_scaled, y_test,
        data_loader.scaler_log_modulus,
        data_loader.scaler_phase_angle,
        device=device
    )
    
    # Add true values for saving
    test_results['true_values'] = y_test
    
    print_evaluation_results(test_results, "Final Model Test Results")
    
    # =========================================================================
    # 6. SENSITIVITY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: SOBOL SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    run_sobol = input("\nPerform Sobol sensitivity analysis? (y/n): ").lower() == 'y'
    
    sobol_results = None
    if run_sobol:
        n_sobol_samples = int(input("Number of Sobol samples (default 1024): ") or "1024")
        
        sobol_results = sobol_sensitivity_analysis(
            best_model,
            split_data['X_train'],
            data_loader.feature_names,
            n_samples=n_sobol_samples,
            scaler_X=data_loader.scaler_X,
            device=device
        )
    
    # =========================================================================
    # 7. SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: SAVING RESULTS")
    print("=" * 80)
    
    # Save model
    model_dir = save_model(
        best_model, model_params,
        data_loader.scaler_X,
        data_loader.scaler_log_modulus,
        data_loader.scaler_phase_angle,
        c_values_dict,
        data_loader.feature_names,
        save_dir='saved_models',
        model_name='mt_pcnn_model'
    )
    
    # Save all results to Excel
    excel_filename = save_results_to_excel(
        'MT_PCNN_Results.xlsx',
        test_results=test_results,
        cv_results=cv_results,
        sobol_results=sobol_results,
        c_values_dict=c_values_dict,
        model_params=model_params,
        optimization_history=optimization_history
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("MT-PCNN TRAINING AND ANALYSIS COMPLETED!")
    print("=" * 80)
    print(f"\nModel saved to: {model_dir}")
    print(f"Results saved to: {excel_filename}")
    
    print("\nSummary of completed tasks:")
    print("  ✓ Estimated implicit c values from data for each mixture type")
    print("  ✓ Removed dependency on external c value parameters")
    print("  ✓ Implemented multiple random restarts to avoid local minima")
    if run_cv:
        print("  ✓ Performed 5-fold cross-validation with detailed results")
    if run_sobol:
        print("  ✓ Conducted Sobol variance-based sensitivity analysis")
    if run_optimization:
        print("  ✓ Performed Bayesian hyperparameter optimization")
    print("  ✓ Saved comprehensive results to Excel file")
    
    print("\nExcel file contains:")
    print("  - Test predictions and errors")
    print("  - Test metrics (R², RMSE, MAE)")
    if run_cv:
        print("  - Cross-validation results (per fold and summary)")
    print("  - Estimated c values for each mixture type")
    if run_sobol:
        print("  - Sobol indices (first-order, total-order, second-order)")
    print("  - Model hyperparameters")
    if run_optimization:
        print("  - Optimization history")
    
    print("\n" + "=" * 80)
    print("Thank you for using MT-PCNN!")
    print("=" * 80)


if __name__ == "__main__":
    main()
