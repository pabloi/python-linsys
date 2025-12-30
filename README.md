# python-linsys

A Python toolbox for linear dynamical system identification, Kalman filtering/smoothing, and EM-based estimation.

This is a Python port of the [MATLAB matlab-linsys toolbox](https://github.com/pabloi/matlab-linsys).

## Features

- **Linear System Representation**: State-space model class with support for:
  - State transition, observation, input, and feedthrough matrices (A, B, C, D)
  - Process and measurement noise covariances (Q, R)
  - Initial state and covariance (x0, P0)
  - System properties (stability, observability, controllability)

- **Kalman Filtering & Smoothing**:
  - Standard Kalman filter with prediction and update steps
  - Rauch-Tung-Striebel (RTS) smoother for optimal state estimation
  - Steady-state Kalman gain computation via DARE
  - Support for missing data (NaN values)
  - Information filter variant

- **System Identification**:
  - EM (Expectation-Maximization) algorithm for parameter estimation
  - Subspace identification (N4SID-style)
  - Multiple random restarts for global optimization
  - Constraints: diagonal matrices, stability enforcement, fixed parameters

- **Utilities**:
  - System simulation with process and measurement noise
  - Log-likelihood computation
  - Model selection criteria (BIC, AIC)
  - System transformations and canonical forms

## Installation

```bash
# Clone the repository
git clone https://github.com/pabloi/python-linsys.git
cd python-linsys

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with visualization support
pip install -e ".[viz]"
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## Quick Start

### Creating a Linear System

```python
import numpy as np
from linsys import LinearSystem

# Define system matrices
A = np.array([[0.9, 0.1], [0.0, 0.8]])
C = np.array([[1.0, 0.0], [0.0, 1.0]])
Q = np.eye(2) * 0.1
R = np.eye(2) * 0.2

# Create system
sys = LinearSystem(A=A, C=C, Q=Q, R=R)

print(sys)
# Linear System
#   States: 2
#   Inputs: 1
#   Outputs: 2
#   Stable: True
```

### Simulating a System

```python
# Simulate 100 time steps
y, x, u = sys.simulate(n_steps=100, noise=True)

# Deterministic simulation
y_det, x_det, _ = sys.simulate(n_steps=100, noise=False)
```

### Kalman Filtering and Smoothing

```python
# Filter observations
x_filt, P_filt, log_lik = sys.filter(y)

# Smooth observations (uses future data)
x_smooth, P_smooth, Pt, log_lik = sys.smooth(y)

# One-step-ahead prediction
y_pred, x_pred = sys.predict(y)
```

### System Identification

```python
# EM identification
from linsys.em import EMOptions

opts = EMOptions(max_iter=100, tol=1e-6, verbose=True)
identified_sys = LinearSystem.identify(y, n_states=2, method="em", opts=opts)

# Subspace identification
identified_sys = LinearSystem.identify(y, n_states=2, method="subspace")

# Random-start EM for better convergence
from linsys.em import random_start_em
best_sys = random_start_em(y, n_states=2, n_restarts=10)
```

### Working with Kalman Functions Directly

```python
from linsys import kalman_filter, kalman_smoother, kalman_predict, kalman_update

# Full Kalman filter
x_filt, P_filt, x_pred, P_pred, log_lik = kalman_filter(
    y, A, C, Q, R, x0=x0, P0=P0
)

# Full Kalman smoother
x_smooth, P_smooth, Pt, x_filt, P_filt, log_lik = kalman_smoother(
    y, A, C, Q, R
)
```

## Examples

### System Identification from Data

```python
import numpy as np
from linsys import LinearSystem
from linsys.em import EMOptions

# Generate data from a true system
rng = np.random.default_rng(42)
true_sys = LinearSystem.random(n_states=3, n_outputs=4, stable=True, rng=rng)
y, x_true, u = true_sys.simulate(n_steps=1000, noise=True, rng=rng)

# Identify the system
opts = EMOptions(max_iter=200, tol=1e-6)
est_sys = LinearSystem.identify(y, n_states=3, method="em", opts=opts)

# Compare eigenvalues (up to rotation)
true_eigs = np.sort(np.abs(true_sys.eigenvalues()))
est_eigs = np.sort(np.abs(est_sys.eigenvalues()))
print(f"True eigenvalues: {true_eigs}")
print(f"Estimated eigenvalues: {est_eigs}")
```

### Handling Missing Data

```python
import numpy as np
from linsys import LinearSystem

# Create system and simulate
sys = LinearSystem.random(n_states=2, n_outputs=2)
y, x, _ = sys.simulate(n_steps=100)

# Introduce missing data
y_missing = y.copy()
y_missing[0, 20:30] = np.nan  # Missing observations

# Filter still works with missing data
x_filt, P_filt, log_lik = sys.filter(y_missing)
```

### Model Comparison

```python
from linsys.utils import bic_aic, count_parameters

# Compare models with different state dimensions
models = []
for n_states in [1, 2, 3, 4]:
    sys = LinearSystem.identify(y, n_states=n_states, method="subspace")
    ll = sys.log_likelihood(y)
    n_params = count_parameters(n_states, sys.n_inputs, sys.n_outputs)
    bic, aic = bic_aic(ll, n_params, y.shape[1] * y.shape[0])
    print(f"States: {n_states}, LL: {ll:.1f}, BIC: {bic:.1f}, AIC: {aic:.1f}")
```

## API Reference

### LinearSystem

Main class for state-space models.

**Constructor:**
- `LinearSystem(A, C, B=None, D=None, Q=None, R=None, x0=None, P0=None, name="")`

**Properties:**
- `n_states`, `n_inputs`, `n_outputs`: Dimensions
- `is_stable()`, `is_observable()`, `is_controllable()`: System properties
- `eigenvalues()`: Eigenvalues of A

**Methods:**
- `simulate(n_steps, u=None, noise=True)`: Simulate the system
- `filter(y, u=None)`: Kalman filter
- `smooth(y, u=None)`: Kalman smoother
- `predict(y, u=None)`: One-step-ahead prediction
- `log_likelihood(y, u=None)`: Compute log-likelihood
- `transform(T)`: Apply similarity transformation
- `reduce(n_states)`: Model order reduction
- `copy()`: Deep copy

**Class Methods:**
- `LinearSystem.random(n_states, n_outputs, n_inputs, stable=True)`: Random system
- `LinearSystem.identify(y, n_states, method="em")`: System identification

### Kalman Functions

- `kalman_filter(y, A, C, Q, R, ...)`: Full Kalman filter
- `kalman_smoother(y, A, C, Q, R, ...)`: RTS smoother
- `kalman_predict(x, P, A, Q, B=None, u=None)`: Prediction step
- `kalman_update(x_pred, P_pred, y, C, R, ...)`: Update step
- `steady_state_kalman_gain(A, C, Q, R)`: Compute steady-state gain

### EM Functions

- `em_identify(y, n_states, ...)`: High-level EM identification
- `em_algorithm(y, n_states, ...)`: Full EM with detailed results
- `em_step(y, A, B, C, D, Q, R, x0, P0, ...)`: Single EM iteration
- `random_start_em(y, n_states, n_restarts, ...)`: Multi-start EM

### Subspace Functions

- `subspace_id(y, n_states, u=None, horizon=10)`: Subspace identification
- `hankel_matrix(data, n_rows, n_cols=None)`: Build Hankel matrix
- `estimate_transition_matrix(X, U=None)`: Estimate A, B from states

### Utility Functions

- `simulate(A, C, n_steps, ...)`: Simulate a system
- `log_likelihood(y, A, C, Q, R, ...)`: Compute log-likelihood
- `bic_aic(log_lik, n_params, n_samples)`: Model selection criteria
- `substitute_nans(y, method="interpolate")`: Handle missing data

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=linsys --cov-report=html

# Run specific test file
pytest tests/test_kalman.py -v
```

## References

- Shumway, R. H., & Stoffer, D. S. (1982). An approach to time series smoothing and forecasting using the EM algorithm. *Journal of Time Series Analysis*, 3(4), 253-264.
- Van Overschee, P., & De Moor, B. (1996). *Subspace Identification for Linear Systems*. Springer.
- Ghahramani, Z., & Hinton, G. E. (1996). Parameter estimation for linear dynamical systems. Technical Report CRG-TR-96-2, University of Toronto.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
