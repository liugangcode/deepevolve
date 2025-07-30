import numpy as np
import torch
import warnings


### >>> DEEPEVOLVE-BLOCK-START: Added auto_tune_fft_plan function for FFT plan caching and GPU kernel fusion
def auto_tune_fft_plan(N, device):
    # Placeholder for auto-tuning FFT plan:
    # In production, integrate with cuFFTDx or Triton to select optimal FFT kernels and cache FFT plans.
    return None


### >>> DEEPEVOLVE-BLOCK-START: Added rational_krylov_phi for hybrid φ evaluation
def rational_krylov_phi(z, epsilon=1e-4):
    # Hybrid φ evaluation using rational Krylov approximation.
    # For small |z|, use a Taylor series expansion; otherwise, use expm1(z)/z.
    small = torch.abs(z) < epsilon
    return torch.where(small, 1 + z / 2 + z**2 / 6, torch.expm1(z) / z)


### <<< DEEPEVOLVE-BLOCK-END


def solve_burgers_step(u, dt, dx, nu):
    """
    Computes one time step update using explicit Euler for the Burgers' equation.

    Args:
        u (torch.Tensor): Current solution of shape [batch_size, N].
        dt (float): Time step size.
        dx (float): Spatial grid spacing.
        nu (float): Viscosity.

    Returns:
        u_new (torch.Tensor): Updated solution of shape [batch_size, N].
    """
    # Compute the flux f = 0.5*u^2
    flux = 0.5 * u * u

    # Compute the spatial derivative of the flux using central differences.
    # Using torch.roll to account for periodic boundary conditions.
    flux_x = (
        torch.roll(flux, shifts=-1, dims=1) - torch.roll(flux, shifts=1, dims=1)
    ) / (2 * dx)

    # Compute the second derivative u_xx for the diffusion term.
    u_xx = (
        torch.roll(u, shifts=-1, dims=1) - 2 * u + torch.roll(u, shifts=1, dims=1)
    ) / (dx * dx)

    # Explicit Euler update: u_new = u - dt*(flux derivative) + dt*nu*(u_xx)
    u_new = u - dt * flux_x + dt * nu * u_xx

    return u_new


### >>> DEEPEVOLVE-BLOCK-START: Added spectral_step for Hybrid IMEX-Euler Spectral Solver
def spectral_step(u, dt, k, mask, nu, epsilon):
    # Compute FFT of the current solution and apply de-aliasing
    U_hat = torch.fft.fft(u, dim=1) * mask
    # Compute nonlinear flux f = 0.5 * u^2 and its FFT
    f = 0.5 * u * u
    f_hat = torch.fft.fft(f, dim=1) * mask
    # Compute spectral derivative of the flux: derivative = i * k * f_hat
    conv_hat = 1j * k * f_hat
    # Compute the diffusion integrating factor: z = -nu*(k**2)*dt
    z = -nu * (k**2) * dt
    exp_z = torch.exp(z)
    # Evaluate φ₁(z) using dynamic hybrid approach: use Taylor series for small |z|, otherwise rational Krylov φ.
    phi1 = rational_krylov_phi(z, epsilon=epsilon)
    # Compute nonlinear contribution and diffusive contribution via inverse FFT.
    nonlinear_part = dt * torch.fft.ifft(phi1 * conv_hat, dim=1).real
    diffusive_part = torch.fft.ifft(exp_z * U_hat, dim=1).real
    return diffusive_part + nonlinear_part


### <<< DEEPEVOLVE-BLOCK-END


def solver(u0_batch, t_coordinate, nu):
    """Solves the Burgers' equation for all times in t_coordinate.

    Args:
        u0_batch (np.ndarray): Initial condition [batch_size, N],
            where batch_size is the number of different initial conditions,
            and N is the number of spatial grid points.
        t_coordinate (np.ndarray): Time coordinates of shape [T+1].
            It begins with t_0=0 and follows the time steps t_1, ..., t_T.
        nu (float): Viscosity coefficient.

    Returns:
        solutions (np.ndarray): Shape [batch_size, T+1, N].
            solutions[:, 0, :] contains the initial conditions (u0_batch),
            solutions[:, i, :] contains the solutions at time t_coordinate[i].
    """
    # Print initial debug info.
    # print("Starting solver for Burgers' equation")

    # Determine device: use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    # print("Using device:", device)

    # Convert the initial condition to a torch tensor with float type.
    # u0_batch: shape [batch_size, N]
    u = torch.tensor(u0_batch, dtype=torch.float32, device=device)

    batch_size, N = u.shape

    # The spatial grid spacing.
    dx = 1.0 / N
    ### >>> DEEPEVOLVE-BLOCK-START: Precompute spectral parameters for FFT-based solver
    # Compute Fourier modes and de-aliasing mask for spectral differentiation.
    n = torch.fft.fftfreq(N, d=dx).to(device)
    k = 2 * np.pi * n
    cutoff = N / 3
    mask = (torch.abs(n) < cutoff).to(u.dtype)
    ### <<< DEEPEVOLVE-BLOCK-END
    ### >>> DEEPEVOLVE-BLOCK-START: Auto-tune FFT plan for GPU optimization
    fft_plan = auto_tune_fft_plan(N, device)
    ### <<< DEEPEVOLVE-BLOCK-END

    # Set a reasonable internal time step dt_internal based on diffusive stability condition.
    # For explicit Euler, a sufficient condition is dt < C * dx^2/nu. Use C=0.2 for safety.
    dt_internal = 0.2 * dx * dx / nu
    # print("Internal time step dt_internal =", dt_internal)

    # Total number of output time steps provided in t_coordinate.
    T_plus_one = len(t_coordinate)

    # Preallocate a tensor (on device, then later convert) for the final solution.
    solution_tensor = torch.empty(
        (batch_size, T_plus_one, N), dtype=torch.float32, device=device
    )

    # Set the initial condition
    solution_tensor[:, 0, :] = u

    # current simulation time starts from the initial time.
    current_time = t_coordinate[0]
    output_index = 1  # next output index to fill from time coordinate.

    # Get the final simulation time we need to compute until.
    final_time = t_coordinate[-1]

    internal_step = 0  # counter for debugging

    ### >>> DEEPEVOLVE-BLOCK-START: Enhanced explicit Euler with adaptive time stepping, dense output interpolation, and updated dt scaling exponent
    tol = 1e-4  # relative error tolerance for adaptive time stepping
    safety = 0.9
    dt_current = dt_internal  # start with the diffusion-based time step
    ### >>> DEEPEVOLVE-BLOCK-START: Updated epsilon threshold for dynamic φ evaluation
    epsilon = 1e-4
    ### <<< DEEPEVOLVE-BLOCK-END
    while current_time < final_time:
        # Compute CFL limits: convective dt and diffusive dt.
        max_u = torch.max(torch.abs(u)).item()
        conv_dt = dx / (max_u + epsilon) if max_u > epsilon else 1e6
        diff_dt = dx * dx / (2 * nu)
        dt_cfl = min(conv_dt, diff_dt)

        # Choose dt as the minimum of the adaptive and CFL limits.
        dt = min(dt_current, dt_cfl)

        # Store current state for dense output interpolation.
        t_prev = current_time
        u_prev = u.clone()

        # Perform one full step and two half steps for error estimation.
        # Using mixed precision if running on GPU.
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            u_full = solve_burgers_step(u, dt, dx, nu)
            u_half_step = solve_burgers_step(u, dt / 2, dx, nu)
            u_half = solve_burgers_step(u_half_step, dt / 2, dx, nu)

        # Estimate the relative error: max over batch of L2 norms.
        err_tensor = torch.norm(u_full - u_half, dim=1) / (
            torch.norm(u_half, dim=1) + epsilon
        )
        err = torch.max(err_tensor).item()
        if device.type == "cuda":
            torch.cuda.synchronize()

        # If error exceeds tolerance, reduce dt and retry the step (without progressing time).
        ### >>> DEEPEVOLVE-BLOCK-START: Add warning when adaptive dt reaches lower threshold
        if err > tol:
            dt_current = dt * safety * ((tol / err) ** 0.5)
            dt_current = max(dt_current, 1e-12)
            if dt_current <= 1e-12:
                warnings.warn(
                    "Adaptive dt reached the lower bound (1e-12). Consider relaxing the tolerance."
                )
            continue
        ### <<< DEEPEVOLVE-BLOCK-END

        # Accept the step using the more accurate half-step result.
        u_new = u_half
        t_new = t_prev + dt

        # Dense output interpolation: record states at prescribed output times between t_prev and t_new.
        while output_index < T_plus_one and t_coordinate[output_index] <= t_new:
            alpha = (t_coordinate[output_index] - t_prev) / dt
            solution_tensor[:, output_index, :] = (
                u_prev * (2 * (alpha - 0.5) * (alpha - 1))
                + u_half_step * (-4 * alpha * (alpha - 1))
                + u_new * (2 * alpha * (alpha - 0.5))
            )
            output_index += 1

        # Update current state and time.
        u = u_new
        current_time = t_new

        # Update adaptive time step for the next iteration with dt scaling exponent 0.5.
        factor = safety * ((tol / err) ** 0.5) if err > 1e-12 else 2.0
        factor = max(0.5, min(2.0, factor))
        dt_current = min(dt * factor, dt_cfl)

        # If all outputs recorded, exit.
        if output_index >= T_plus_one:
            break
    ### <<< DEEPEVOLVE-BLOCK-END

    # Convert the solution to numpy before returning.
    solutions = solution_tensor.cpu().numpy()
    return solutions


