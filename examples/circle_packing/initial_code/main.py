"""Constructor-based circle packing for n=26 circles"""

import numpy as np
from time import time
import traceback
from scipy.optimize import minimize


def construct_packing(n=26):
    """
    Compute circle packing for n circles in the unit square using multiple SLSQP restarts.
    Returns:
        centers: array of shape (n, 2)
        radii: array of shape (n,)
        sum_radii: float
    """
    # Prebuild bounds and constraints
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    constraints = []

    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):

            def overlap(x, i=i, j=j):
                xi, yi = x[2 * i], x[2 * i + 1]
                xj, yj = x[2 * j], x[2 * j + 1]
                ri = x[2 * n + i]
                rj = x[2 * n + j]
                dist = np.hypot(xi - xj, yi - yj)
                return dist - (ri + rj)

            constraints.append({"type": "ineq", "fun": overlap})

    # Boundary constraints
    for i in range(n):

        def left(x, i=i):
            return x[2 * i] - x[2 * n + i]

        def right(x, i=i):
            return 1 - (x[2 * i] + x[2 * n + i])

        def bottom(x, i=i):
            return x[2 * i + 1] - x[2 * n + i]

        def top(x, i=i):
            return 1 - (x[2 * i + 1] + x[2 * n + i])

        constraints.extend(
            [
                {"type": "ineq", "fun": left},
                {"type": "ineq", "fun": right},
                {"type": "ineq", "fun": bottom},
                {"type": "ineq", "fun": top},
            ]
        )

    best_sum = -np.inf
    best_x = None

    rng = np.random.default_rng(42)
    centers0 = rng.uniform(0.1, 0.9, size=(n, 2))
    radii0 = np.full(n, 0.05)
    x0 = np.hstack((centers0.flatten(), radii0))

    def objective(x):
        return -np.sum(x[2 * n :])

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-6},
    )

    if result.success:
        radii = result.x[2 * n :]
        total = np.sum(radii)
        if total > best_sum:
            best_sum = total
            best_x = result.x.copy()

    if best_x is None:
        return [], [], 0.0

    centers = best_x[: 2 * n].reshape(n, 2)
    radii = best_x[2 * n :]
    return centers, radii, best_sum

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            message = (
                f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square"
            )
            return False, message

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j]:
                message = f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}"
                return False, message

    return True, "success"

def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()
    plt.savefig('circle_packing.png')

if __name__ == "__main__":
    centers, radii, sum_radii = construct_packing(n=28)
    print('centers', centers)
    print('radii', radii)
    print('sum_radii', sum_radii)

    valid_packing, message_packing = validate_packing(centers, radii)
    print('valid_packing', valid_packing)
    print('message_packing', message_packing)

    # visualize(centers, radii)