# PDE Burgers Solver

This repository contains a solver for the one-dimensional viscous Burgers equation. The solver is tailored for the specific case where the viscosity $\nu = \text{burgers\_nu}$, and it is optimized for this use. The code is implemented in PyTorch and leverages GPU acceleration when available.

---

## Problem Description

We aim to solve the following partial differential equation (PDE):

```math
\begin{cases}
\partial_t u(x, t) + \partial_x \Bigl(\frac{u^2(x, t)}{2}\Bigr) = \nu\, \partial_{xx} u(x, t), & x \in (0,1), \; t \in (0,1] \\
u(x, 0) = u_0(x), & x \in (0,1)
\end{cases}
```

with periodic boundary conditions. The initial condition $u_0(x)$ is provided as a discretized array with shape `[batch_size, N]`, where $N$ is the number of spatial points. The goal is to predict the evolution of $u(\cdot, t)$ at specified time steps $t = t_1, \dots, t_T$, producing an output of shape `[batch_size, T+1, N]` (including the initial condition).

**Note:** To ensure numerical stability, the solver may use smaller internal time steps than those specified for the output.

---

## Evaluation Metrics

The performance of the solver is measured using the following metrics:

1. **Scale-Independent Normalized Root Mean Squared Error (nRMSE):**

   For a set of $S$ PDE examples, the nRMSE is defined as:
   
   ```math
   \text{nRMSE} = \frac{1}{S} \sum_{s=1}^{S} \frac{\| u^{(s)}(x,t) - \hat{u}^{(s)}(x,t) \|_{2}}{\| u^{(s)}(x,t) \|_{2}}
   ```
   
   where $u^{(s)}(x,t)$ is the ground truth and $\hat{u}^{(s)}(x,t)$ is the predicted solution.

2. **Convergence Rate:**

   The convergence test assesses if the solution error decreases as the grid is refined. Specifically, for a grid spacing $h$, the solver is considered convergent if:
   
   ```math
   \| u_{h} - u_{h/2} \|_{2} \rightarrow 0 \quad \text{as} \quad h \rightarrow 0.
   ```
   
   This ensures the numerical solution approaches the reference solution at the expected rate, confirming consistency and correctness.

3. **Computational Efficiency:**

   The execution time of the solver is recorded to measure its computational efficiency.

---

## Solver Interface

The solver is implemented in the file `deepevolve_interface.py`. This interface defines the structure and methods for interacting with the solver and is designed to integrate with the broader system.

---

## Initial Idea

The initial idea behind the solver is as follows:

- **Equation:** The solver integrates the one-dimensional viscous Burgers equation:
  
  ```math
  u_t + \frac{1}{2}(u^2)_x = \nu\, u_{xx}
  ```
  
- **Spatial Discretization:** For each batch of $B$ initial states sampled on an evenly spaced grid of $N$ points (with $\Delta x = 1/N$):
  
  - Compute the convective flux $f = \frac{1}{2}u^2$.
  - Evaluate the spatial derivative of the convective flux using a centered finite-difference stencil implemented through `torch.roll`.
  - Compute the diffusion term $u_{xx}$ using the standard three-point Laplacian.

- **Time Integration:** 
  
  - The solver uses an explicit Euler method for time integration.
  - The time step for the inner loop is chosen adaptively but never exceeds $0.2\,\Delta x^2/\nu$, which satisfies the explicit stability criterion for the diffusive term.
  - The simulation is advanced on the GPU (when available), updating the solution tensor in place until the simulation time matches each requested output time provided by the user in the array $\{t_0, \dots, t_T\}$.
  - At each specified output time, the current field is stored, resulting in a final output tensor of shape `[B, T+1, N]` in single precision before conversion back to NumPy format.

For a more detailed implementation, please refer to the supplementary material:

[Supplementary Implementation](https://github.com/LithiumDA/CodePDE/blob/main/solvers/burgers/nu_1.0/seeds/implementation_0.py)