"""
Data Loading and Preprocessing Module

This module handles data loading from Excel files, preprocessing, normalization,
and train-test splitting for the MT-PCNN model.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_log_modulus = StandardScaler()
        self.scaler_phase_angle = StandardScaler()
        self.feature_names = None
        
        # Data arrays
        self.mixture_ids = None
        self.dynamic_modulus = None
        self.phase_angle = None
        self.omega = None
        self.log_omega = None
        self.log_modulus = None
        self.mixture_properties = None
        self.X = None
        self.y = None
    
    def load_data(self, file_path='Ex_data.xlsx', sheet_name='Sheet1'):

        try:
            # Load main data from Excel
            main_data = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Extract columns
            self.mixture_ids = main_data.iloc[:, 0].values
            self.dynamic_modulus = main_data.iloc[:, 1].values  # psi
            self.phase_angle = main_data.iloc[:, 2].values      # rad
            self.omega = main_data.iloc[:, 3].values            # rad/s
            self.mixture_properties = main_data.iloc[:, 4:].values
            
            # Store feature names
            self.feature_names = ['log_omega'] + list(main_data.columns[4:])
            
            # Convert to log scale
            self.log_modulus = np.log10(self.dynamic_modulus)
            self.log_omega = np.log10(self.omega)
            
            # Create input features (log_omega + mixture properties)
            self.X = np.column_stack([
                self.log_omega.reshape(-1, 1),
                self.mixture_properties
            ])
            
            # Create target matrix
            self.y = np.column_stack([self.log_modulus, self.phase_angle])
            
            print(f"\nData loaded successfully from {file_path}")
            print(f"Total samples: {len(self.X)}")
            print(f"Number of features: {self.X.shape[1]}")
            print(f"Number of unique mixtures: {len(np.unique(self.mixture_ids))}")
            print(f"Feature names: {self.feature_names}")
            
            return {
                'X': self.X,
                'y': self.y,
                'mixture_ids': self.mixture_ids,
                'log_omega': self.log_omega,
                'log_modulus': self.log_modulus,
                'phase_angle': self.phase_angle
            }
            
        except Exception as e:
            raise ValueError(f"Error loading data from {file_path}: {e}")
    
    def prepare_data(self, test_size=0.2, random_state=42):

        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Split data
        indices = np.arange(len(self.X))
        
        (X_train, X_test, 
         y_train, y_test,
         train_indices, test_indices,
         train_mixture_ids, test_mixture_ids) = train_test_split(
            self.X, self.y, indices, self.mixture_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=None  # Could stratify by mixture_id if needed
        )
        
        # Fit scalers on training data
        self.scaler_X.fit(X_train)
        self.scaler_log_modulus.fit(y_train[:, 0].reshape(-1, 1))
        self.scaler_phase_angle.fit(y_train[:, 1].reshape(-1, 1))
        
        # Transform data
        X_train_scaled = self.scaler_X.transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_log_scaled = self.scaler_log_modulus.transform(
            y_train[:, 0].reshape(-1, 1)
        )
        y_test_log_scaled = self.scaler_log_modulus.transform(
            y_test[:, 0].reshape(-1, 1)
        )
        
        y_train_phase_scaled = self.scaler_phase_angle.transform(
            y_train[:, 1].reshape(-1, 1)
        )
        y_test_phase_scaled = self.scaler_phase_angle.transform(
            y_test[:, 1].reshape(-1, 1)
        )
        
        y_train_scaled = np.column_stack([
            y_train_log_scaled.ravel(),
            y_train_phase_scaled.ravel()
        ])
        y_test_scaled = np.column_stack([
            y_test_log_scaled.ravel(),
            y_test_phase_scaled.ravel()
        ])
        
        print(f"\nData split into train/test sets:")
        print(f"Training samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
        print(f"Test samples: {len(X_test)} ({test_size*100:.0f}%)")
        
        return {
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train_scaled': y_train_scaled,
            'y_test_scaled': y_test_scaled,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'train_mixture_ids': train_mixture_ids,
            'test_mixture_ids': test_mixture_ids
        }
    
    def inverse_transform_predictions(self, predictions_scaled):

        log_modulus = self.scaler_log_modulus.inverse_transform(
            predictions_scaled[:, 0].reshape(-1, 1)
        ).ravel()
        
        phase_angle = self.scaler_phase_angle.inverse_transform(
            predictions_scaled[:, 1].reshape(-1, 1)
        ).ravel()
        
        return np.column_stack([log_modulus, phase_angle])
