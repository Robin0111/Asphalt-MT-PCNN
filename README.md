
# MT-PCNN: Multi-Task Physics-Constrained Neural Network
A multi-task physics-constrained neural network for predicting dynamic modulus and phase angle of asphalt mixtures using Kramers-Kronig relations as physical constraints.

## Overview

This repository implements a Multi-Task Physics-Constrained Neural Network (MT-PCNN) that simultaneously predicts:
- *Dynamic Modulus (|E*|)*: Material stiffness under dynamic loading
- *Phase Angle*: Viscoelastic phase lag

The model incorporates Kramers-Kronig relations as physics-based constraints to ensure thermodynamic consistency between predictions.

### Key Features

- **Physics-Constrained Learning**: Incorporates Kramers-Kronig relations as soft constraints
- **Implicit Parameter Estimation**: Automatically estimates material-specific constants from data
- **Random Restart Strategy**: Avoids local minima through multiple training initializations
- **Bayesian Optimization**: Automated hyperparameter tuning using Gaussian Processes
- **Comprehensive Analysis**: Includes cross-validation and Sobol sensitivity analysis
- **Modular Design**: Clean, well-documented, and easy to extend

## Repository Structure

```
mt-pcnn/
├── main.py                      # Main pipeline script
├── models.py                    # Neural network architecture
├── physics_constraints.py       # Kramers-Kronig relations and c-value estimation
├── data_loader.py              # Data loading and preprocessing
├── trainer.py                   # Training logic with random restarts
├── evaluation.py                # Model evaluation and cross-validation
├── sensitivity_analysis.py      # Sobol sensitivity analysis
├── optimization.py              # Bayesian hyperparameter optimization
├── utils.py                     # Saving/loading utilities
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mt-pcnn.git
cd mt-pcnn
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Format

The model expects an Excel file (`EX_data.xlsx`):
**Column Descriptions:**
- **Column 0**: Mixture identifier (for grouping samples)
- **Column 1**: Dynamic modulus in psi
- **Column 2**: Phase angle in radians
- **Column 3**: Angular frequency in rad/s
- **Columns 4+**: Mixture properties (e.g., binder content, air voids, temperature)


```
MT-PCNN: Multi-Task Physics-Constrained Neural Network
================================================================================

### Module Usage

You can also use individual modules in your own scripts:

```python
from data_loader import DataLoader
from physics_constraints import estimate_c_values
from trainer import Trainer
from models import PCNNModel


## Physics Background

### Kramers-Kronig Relations

For linear viscoelastic materials, the Kramers-Kronig relations establish a fundamental connection between the dynamic modulus and phase angle:
φ(ω) = (π/2) · c · d(log|E*|)/d(log(ω))
where:
- **φ(ω)**: Phase angle as a function of frequency
- **|E*|**: Dynamic modulus magnitude
- **ω**: Angular frequency
- **c**: Material-specific scaling coefficient

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


