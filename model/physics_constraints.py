"""
Physics Constraints Module

This module implements the Kramers-Kronig relations as physics-based constraints
for the neural network training. These relations provide thermodynamic consistency
between the dynamic modulus and phase angle of viscoelastic materials.
"""

import torch
import numpy as np


def estimate_c_values(mixture_ids, log_omega, log_modulus, phase_angle):

    unique_mixtures = np.unique(mixture_ids)
    c_values_dict = {}
    
    print("\nEstimating implicit c values for each mixture type:")
    print("-" * 60)
    
    for mixture_id in unique_mixtures:
        # Get data for this mixture
        mask = mixture_ids == mixture_id
        omega_mix = log_omega[mask]
        modulus_mix = log_modulus[mask]
        phase_mix = phase_angle[mask]
        
        # Sort by frequency for proper derivative calculation
        sort_idx = np.argsort(omega_mix)
        omega_sorted = omega_mix[sort_idx]
        modulus_sorted = modulus_mix[sort_idx]
        phase_sorted = phase_mix[sort_idx]
        
        # Calculate numerical derivative d(log|E*|)/d(log(ω))
        d_modulus_d_omega = _compute_derivative(omega_sorted, modulus_sorted)
        
        # Calculate c values: c = φ / [(π/2) * d(log|E*|)/d(log(ω))]
        # Avoid division by very small derivatives
        valid_mask = np.abs(d_modulus_d_omega) > 1e-6
        
        if np.sum(valid_mask) > 0:
            c_estimates = phase_sorted[valid_mask] / \
                         ((np.pi / 2) * d_modulus_d_omega[valid_mask])
            
            # Use median to be robust against outliers
            # Clip to reasonable range [0.5, 1.5] based on physical constraints
            c_value = np.clip(np.median(c_estimates), 0.5, 1.5)
        else:
            # Default value if calculation fails
            c_value = 1.0
            print(f"  Warning: Using default c=1.0 for mixture {mixture_id}")
        
        c_values_dict[mixture_id] = c_value
        print(f"  Mixture {mixture_id}: c = {c_value:.4f} "
              f"(based on {np.sum(valid_mask)} valid points)")
    
    print("-" * 60)
    print(f"Average c value across all mixtures: {np.mean(list(c_values_dict.values())):.4f}")
    print(f"C value range: [{min(c_values_dict.values()):.4f}, "
          f"{max(c_values_dict.values()):.4f}]")
    
    return c_values_dict


def _compute_derivative(x, y):

    derivative = np.zeros_like(y)
    
    if len(x) > 2:
        # Central differences for interior points
        for i in range(1, len(x) - 1):
            derivative[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
        
        # Forward difference for first point
        derivative[0] = (y[1] - y[0]) / (x[1] - x[0])
        
        # Backward difference for last point
        derivative[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    elif len(x) == 2:
        # Simple slope for two points
        slope = (y[1] - y[0]) / (x[1] - x[0])
        derivative[:] = slope
    
    return derivative


def kramers_kronig_loss(predictions, inputs, c_values_tensor):

    log_modulus_pred = predictions[:, 0]
    phase_angle_pred = predictions[:, 1]
    log_omega = inputs[:, 0]
    
    # Enable gradient computation for log_modulus with respect to log_omega
    log_omega_tensor = log_omega.clone().detach().requires_grad_(True)
    
    # Create modified input tensor for gradient computation
    inputs_grad = inputs.clone()
    inputs_grad[:, 0] = log_omega_tensor
    
    # Recompute predictions with gradient tracking
    log_modulus_grad = predictions[:, 0].clone()
    
    # Compute gradient d(log|E*|)/d(log(ω))
    gradients = torch.autograd.grad(
        outputs=log_modulus_grad,
        inputs=log_omega_tensor,
        grad_outputs=torch.ones_like(log_modulus_grad),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Kramers-Kronig relation: φ = (π/2) * c * d(log|E*|)/d(log(ω))
    phase_angle_kk = (np.pi / 2) * c_values_tensor * gradients
    
    # Calculate loss (MSE between predicted and K-K derived phase angles)
    kk_loss = torch.mean((phase_angle_pred - phase_angle_kk) ** 2)
    
    return kk_loss
