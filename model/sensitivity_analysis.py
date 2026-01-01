"""
Sensitivity Analysis Module

This module implements Sobol variance-based global sensitivity analysis
to quantify the influence of input features on model predictions.
"""

import numpy as np
import torch
from SALib.sample import saltelli
from SALib.analyze import sobol


def sobol_sensitivity_analysis(model, X, feature_names, n_samples=1024, 
                               scaler_X=None, device='cpu'):

    print(f"\nPerforming Sobol sensitivity analysis...")
    print(f"Generating {n_samples * (2 * len(feature_names) + 2)} samples")
    print("-" * 60)
    
    model.eval()
    device = torch.device(device)
    
    # Define the problem for SALib
    problem = {
        'num_vars': len(feature_names),
        'names': feature_names,
        'bounds': [[X[:, i].min(), X[:, i].max()] for i in range(X.shape[1])]
    }
    
    # Generate samples using Saltelli's scheme
    param_values = saltelli.sample(problem, n_samples, calc_second_order=True)
    
    print(f"Generated {len(param_values)} parameter combinations")
    print("Running model predictions...")
    
    # Scale samples if scaler provided
    if scaler_X is not None:
        param_values_scaled = scaler_X.transform(param_values)
    else:
        param_values_scaled = param_values
    
    # Run model predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(param_values_scaled).to(device)
        predictions = model(X_tensor).cpu().numpy()
    
    # Separate predictions for each output
    Y_log_modulus = predictions[:, 0]
    Y_phase_angle = predictions[:, 1]
    
    print("Analyzing Sobol indices...")
    
    # Perform Sobol analysis for each output
    Si_log = sobol.analyze(
        problem, Y_log_modulus, 
        calc_second_order=True,
        print_to_console=False
    )
    
    Si_phase = sobol.analyze(
        problem, Y_phase_angle,
        calc_second_order=True,
        print_to_console=False
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("SOBOL SENSITIVITY ANALYSIS RESULTS")
    print("=" * 80)
    
    print("\nLog Dynamic Modulus Sensitivity:")
    print("-" * 80)
    _print_sobol_results(Si_log, feature_names)
    
    print("\nPhase Angle Sensitivity:")
    print("-" * 80)
    _print_sobol_results(Si_phase, feature_names)
    
    return {
        'log_modulus': Si_log,
        'phase_angle': Si_phase,
        'feature_names': feature_names
    }


def _print_sobol_results(Si, feature_names):

    print(f"\n{'Feature':<25} {'S1 (Main)':<15} {'ST (Total)':<15} {'Interaction':<15}")
    print("-" * 70)
    
    for i, name in enumerate(feature_names):
        s1 = Si['S1'][i]
        st = Si['ST'][i]
        interaction = st - s1
        
        print(f"{name:<25} {s1:>14.4f} {st:>14.4f} {interaction:>14.4f}")
    
    print("-" * 70)
    print(f"\nInterpretation:")
    print(f"  S1 (First-order): Direct effect of the variable")
    print(f"  ST (Total-order): Total effect including all interactions")
    print(f"  Interaction = ST - S1: Effect due to interactions with other variables")
    print(f"\nHigher values indicate greater influence on the output.")


def extract_sobol_dataframes(sobol_results):

    import pandas as pd
    
    feature_names = sobol_results['feature_names']
    Si_log = sobol_results['log_modulus']
    Si_phase = sobol_results['phase_angle']
    
    # First-order and total-order indices
    indices_log = pd.DataFrame({
        'Feature': feature_names,
        'S1 (Main Effect)': Si_log['S1'],
        'S1_conf': Si_log['S1_conf'],
        'ST (Total Effect)': Si_log['ST'],
        'ST_conf': Si_log['ST_conf'],
        'Interaction (ST-S1)': Si_log['ST'] - Si_log['S1']
    })
    
    indices_phase = pd.DataFrame({
        'Feature': feature_names,
        'S1 (Main Effect)': Si_phase['S1'],
        'S1_conf': Si_phase['S1_conf'],
        'ST (Total Effect)': Si_phase['ST'],
        'ST_conf': Si_phase['ST_conf'],
        'Interaction (ST-S1)': Si_phase['ST'] - Si_phase['S1']
    })
    
    # Second-order indices (pairwise interactions)
    if 'S2' in Si_log:
        n_features = len(feature_names)
        s2_log_data = []
        s2_phase_data = []
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                s2_log_data.append({
                    'Feature_i': feature_names[i],
                    'Feature_j': feature_names[j],
                    'S2': Si_log['S2'][i, j],
                    'S2_conf': Si_log['S2_conf'][i, j]
                })
                s2_phase_data.append({
                    'Feature_i': feature_names[i],
                    'Feature_j': feature_names[j],
                    'S2': Si_phase['S2'][i, j],
                    'S2_conf': Si_phase['S2_conf'][i, j]
                })
        
        second_order_log = pd.DataFrame(s2_log_data)
        second_order_phase = pd.DataFrame(s2_phase_data)
    else:
        second_order_log = None
        second_order_phase = None
    
    return {
        'first_order_log': indices_log,
        'total_order_log': indices_log[['Feature', 'ST (Total Effect)', 'ST_conf']],
        'first_order_phase': indices_phase,
        'total_order_phase': indices_phase[['Feature', 'ST (Total Effect)', 'ST_conf']],
        'second_order_log': second_order_log,
        'second_order_phase': second_order_phase
    }
