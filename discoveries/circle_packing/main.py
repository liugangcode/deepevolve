"""Constructor-based circle packing for n=26 circles"""

import numpy as np
from time import time
import traceback
from scipy.optimize import minimize


# DEBUG: added stub for interval arithmetic verification
def interval_verification(x, n):
    """
    Interval arithmetic based verification of circle packing.
    Stub implementation using validate_packing.
    """
    # x: concatenated [centers.flatten(), radii]
    centers = np.array(x[: 2 * n]).reshape(n, 2)
    radii = np.array(x[2 * n :])
    valid, _ = validate_packing(centers, radii)
    return valid


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

    # Non-overlap constraints with analytic gradients
    def non_overlap_gradient(x, i, j):
        xi, yi = x[2 * i], x[2 * i + 1]
        xj, yj = x[2 * j], x[2 * j + 1]
        diff = np.array([xi - xj, yi - yj])
        d = np.hypot(diff[0], diff[1]) + 1e-10
        grad = np.zeros_like(x)
        grad[2 * i] = diff[0] / d
        grad[2 * i + 1] = diff[1] / d
        grad[2 * j] = -diff[0] / d
        grad[2 * j + 1] = -diff[1] / d
        grad[2 * n + i] = -1
        grad[2 * n + j] = -1
        return grad

    for i in range(n):
        for j in range(i + 1, n):

            def overlap(x, i=i, j=j):
                xi, yi = x[2 * i], x[2 * i + 1]
                xj, yj = x[2 * j], x[2 * j + 1]
                ri = x[2 * n + i]
                rj = x[2 * n + j]
                dist = np.hypot(xi - xj, yi - yj)
                return dist - (ri + rj)

            def overlap_jac(x, i=i, j=j):
                return non_overlap_gradient(x, i, j)

            constraints.append({"type": "ineq", "fun": overlap, "jac": overlap_jac})

    # Boundary constraints with analytic gradients
    def jac_left(x, i):
        grad = np.zeros_like(x)
        grad[2 * i] = 1
        grad[2 * n + i] = -1
        return grad

    def jac_right(x, i):
        grad = np.zeros_like(x)
        grad[2 * i] = -1
        grad[2 * n + i] = -1
        return grad

    def jac_bottom(x, i):
        grad = np.zeros_like(x)
        grad[2 * i + 1] = 1
        grad[2 * n + i] = -1
        return grad

    def jac_top(x, i):
        grad = np.zeros_like(x)
        grad[2 * i + 1] = -1
        grad[2 * n + i] = -1
        return grad

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
                {"type": "ineq", "fun": left, "jac": lambda x, i=i: jac_left(x, i)},
                {"type": "ineq", "fun": right, "jac": lambda x, i=i: jac_right(x, i)},
                {"type": "ineq", "fun": bottom, "jac": lambda x, i=i: jac_bottom(x, i)},
                {"type": "ineq", "fun": top, "jac": lambda x, i=i: jac_top(x, i)},
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

    def objective_jac(x):
        grad = np.zeros_like(x)
        grad[2 * n :] = -1
        return grad

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
        # DEBUG: no valid SLSQP result, fallback to last optimization output to proceed with refinement
        best_x = result.x.copy()
        best_sum = np.sum(best_x[2 * n :])
        print(
            f"Warning: No valid candidate found for circle packing for n={n}, proceeding with fallback solution."
        )

    centers = best_x[: 2 * n].reshape(n, 2)
    radii = best_x[2 * n :]
    print(f"Multi-start candidate selected with total radii = {best_sum:.6f}")

    # Iterative refinement using power diagram and maximum inscribed circles
    for _ in range(10):
        cells = compute_power_cells(centers, radii)
        new_centers = []
        new_radii = []
        for i, cell in enumerate(cells):
            if cell.is_empty:
                new_centers.append(centers[i])
                new_radii.append(radii[i] * 0.9)
            else:
                point, r_val = find_max_inscribed_circle(cell, resolution=0.002)
                if point is None:
                    new_centers.append(centers[i])
                    new_radii.append(radii[i])
                else:
                    new_centers.append([point.x, point.y])
                    new_radii.append(min(r_val, radii[i] + 0.001))
        new_centers = np.array(new_centers)
        new_radii = np.array(new_radii)
        if (
            np.linalg.norm(new_centers - centers) < 1e-4
            and np.linalg.norm(new_radii - radii) < 1e-4
        ):
            centers, radii = new_centers, new_radii
            break
        centers, radii = new_centers, new_radii

    # Final refinement with SLSQP to enforce non-overlap and boundary constraints
    x0 = np.hstack((centers.flatten(), radii))
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=objective_jac,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-8},
    )
    if result.success:
        radii = result.x[2 * n :]
        centers = result.x[: 2 * n].reshape(n, 2)
        best_sum = np.sum(radii)
    # If the final solution is invalid, apply adaptive perturbation and re-optimize
    valid, msg = validate_packing(centers, radii)
    if not valid:
        max_adaptive_iter = 5
        iteration = 0
        x_candidate = np.hstack((centers.flatten(), radii))
        while (
            not valid or not interval_verification(x_candidate, n)
        ) and iteration < max_adaptive_iter:
            x_candidate = adaptive_perturbation(
                x_candidate, n, scale=0.01 * (iteration + 1)
            )
            result = minimize(
                objective,
                x_candidate,
                method="SLSQP",
                jac=objective_jac,
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-8},
            )
            if result.success:
                x_candidate = result.x.copy()
            centers = x_candidate[: 2 * n].reshape(n, 2)
            radii = x_candidate[2 * n :]
            valid, msg = validate_packing(centers, radii)
            iteration += 1
        if not valid:
            print(
                "Warning: adaptive perturbation failed; falling back to adaptive bisection"
            )
            radii = adaptive_bisection(centers, radii)
            x_candidate = np.hstack((centers.flatten(), radii))
            result = minimize(
                objective,
                x_candidate,
                method="SLSQP",
                jac=objective_jac,
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-8},
            )
            if result.success:
                x_candidate = result.x.copy()
                centers = x_candidate[: 2 * n].reshape(n, 2)
                radii = x_candidate[2 * n :]
                best_sum = np.sum(radii)
    
    return centers, radii, best_sum


# DEBUG: added missing compute_power_cells and find_max_inscribed_circle implementations
### <<< DEEPEVOLVE-BLOCK-END
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import split


def compute_power_cells(centers, radii):
    """
    Compute power cells (weighted Voronoi) for given centers and radii inside the unit square.
    Returns a list of shapely Polygon objects representing each cell.
    """
    # build a large bounding box for half‐space intersections
    M = 10.0
    bb = Polygon([(-M, -M), (M, -M), (M, M), (-M, M)])
    # start from the unit square
    domain = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    cells = []
    n = len(centers)
    for i in range(n):
        poly = domain
        cx_i, cy_i = centers[i]
        weight_i = cx_i * cx_i + cy_i * cy_i - radii[i] * radii[i]
        for j in range(n):
            if j == i:
                continue
            cx_j, cy_j = centers[j]
            weight_j = cx_j * cx_j + cy_j * cy_j - radii[j] * radii[j]
            # half‐space: 2*(c_j - c_i)⋅x <= weight_j - weight_i
            a = 2 * (cx_j - cx_i)
            b = 2 * (cy_j - cy_i)
            c = weight_j - weight_i
            # build splitting line across the big box
            if abs(b) > abs(a) and b != 0:
                p1 = Point(-M, (c - a * (-M)) / b)
                p2 = Point(M, (c - a * (M)) / b)
            else:
                # vertical line (avoid division by zero)
                if a == 0:
                    poly = Polygon()
                    break
                p1 = Point(c / a, -M)
                p2 = Point(c / a, M)
            line = LineString([p1, p2])
            # split the bounding box into two half‐spaces
            # DEBUG: shapely.ops.split returns a GeometryCollection, which is not directly iterable; iterate over pieces.geoms
            pieces = split(bb, line)
            halfspace = None
            for piece in pieces.geoms:
                test_pt = piece.representative_point()
                if a * test_pt.x + b * test_pt.y <= c:
                    halfspace = piece
                    break
            if halfspace is None:
                poly = Polygon()
                break
            poly = poly.intersection(halfspace)
            if poly.is_empty:
                break
        cells.append(poly)
    return cells


def find_max_inscribed_circle(polygon, resolution=0.002):
    """
    Approximate the maximum inscribed circle in a polygon by grid sampling.
    Returns (Point center, radius) or (None, 0) if the polygon is empty.
    """
    if polygon.is_empty:
        return None, 0.0
    minx, miny, maxx, maxy = polygon.bounds
    best_pt = None
    best_r = 0.0
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            pt = Point(x, y)
            if polygon.contains(pt):
                # distance to the boundary
                d = polygon.boundary.distance(pt)
                if d > best_r:
                    best_r = d
                    best_pt = pt
            y += resolution
        x += resolution
    if best_pt is None:
        return None, 0.0
    return best_pt, best_r


### >>> DEEPEVOLVE-BLOCK-START: Adaptive Bisection for Radii Adjustment
def adaptive_bisection(centers, radii, tol=1e-4, max_iter=10):
    """
    Adaptively scale down the radii until the packing becomes valid.
    If after max_iter a valid configuration is not reached, a warning is issued.
    """
    for iteration in range(max_iter):
        valid, msg = validate_packing(centers, radii)
        if valid:
            return radii
        radii = radii * 0.95
    warnings.warn(
        f"adaptive_bisection did not achieve a valid configuration after {max_iter} iterations. Returning last radii."
    )
    return radii


### <<< DEEPEVOLVE-BLOCK-END


### >>> DEEPEVOLVE-BLOCK-START: Adaptive Perturbation Function
def adaptive_perturbation(x, n, scale=0.01):
    """
    Apply an adaptive perturbation to candidate configuration x.
    x is a vector of length 3*n (first 2*n entries are centers, next n entries are radii).
    The function perturbs centers (and slightly adjusts radii) to reduce overlaps
    and enforce boundary clearance.
    """
    centers = x[: 2 * n].reshape(n, 2)
    radii = x[2 * n :]
    new_centers = centers.copy()
    new_radii = radii.copy()
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.hypot(diff[0], diff[1])
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                if dist < 1e-8:
                    direction = np.random.uniform(-1, 1, size=2)
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction /= norm
                    else:
                        direction = np.array([1.0, 0.0])
                else:
                    direction = diff / dist
                perturbation = scale * overlap * direction
                new_centers[i] += perturbation
                new_centers[j] -= perturbation
        if new_centers[i, 0] < radii[i]:
            new_centers[i, 0] = radii[i] + scale
        if new_centers[i, 0] > 1 - radii[i]:
            new_centers[i, 0] = 1 - radii[i] - scale
        if new_centers[i, 1] < radii[i]:
            new_centers[i, 1] = radii[i] + scale
        if new_centers[i, 1] > 1 - radii[i]:
            new_centers[i, 1] = 1 - radii[i] - scale
        total_overlap = 0.0
        for j in range(n):
            if i == j:
                continue
            diff = centers[i] - centers[j]
            dist = np.hypot(diff[0], diff[1])
            total_overlap += max(0, radii[i] + radii[j] - dist)
        if total_overlap > 0:
            new_radii[i] = new_radii[i] * (1 - 0.01 * total_overlap)
    return np.hstack((new_centers.flatten(), new_radii))


### <<< DEEPEVOLVE-BLOCK-END
### >>> DEEPEVOLVE-BLOCK-START: Adaptive Perturbation Function
def adaptive_perturbation(x, n, scale=0.01):
    """
    Apply an adaptive perturbation to a candidate configuration x.
    x is a vector of length 3*n (first 2*n entries are centers, next n entries are radii).
    The function perturbs centers (and slightly adjusts radii) to reduce overlaps
    and enforce boundary clearance.
    """
    centers = x[: 2 * n].reshape(n, 2)
    radii = x[2 * n :]
    new_centers = centers.copy()
    new_radii = radii.copy()
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.hypot(diff[0], diff[1])
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                if dist < 1e-8:
                    direction = np.random.uniform(-1, 1, size=2)
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction /= norm
                    else:
                        direction = np.array([1.0, 0.0])
                else:
                    direction = diff / dist
                perturbation = scale * overlap * direction
                new_centers[i] += perturbation
                new_centers[j] -= perturbation
        if new_centers[i, 0] < radii[i]:
            new_centers[i, 0] = radii[i] + scale
        if new_centers[i, 0] > 1 - radii[i]:
            new_centers[i, 0] = 1 - radii[i] - scale
        if new_centers[i, 1] < radii[i]:
            new_centers[i, 1] = radii[i] + scale
        if new_centers[i, 1] > 1 - radii[i]:
            new_centers[i, 1] = 1 - radii[i] - scale
        total_overlap = 0.0
        for j in range(n):
            if i == j:
                continue
            diff = centers[i] - centers[j]
            dist = np.hypot(diff[0], diff[1])
            total_overlap += max(0, radii[i] + radii[j] - dist)
        if total_overlap > 0:
            new_radii[i] = new_radii[i] * (1 - 0.01 * total_overlap)
    return np.hstack((new_centers.flatten(), new_radii))


### <<< DEEPEVOLVE-BLOCK-END
def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square.

    Args:
        centers: np.array of shape (n, 2) containing (x, y) coordinates.
        radii: np.array of shape (n,) with the radius of each circle.

    Returns:
        (bool, str): Tuple where the first element is True if valid, False otherwise,
        and the second element is a message.
    """
    n = centers.shape[0]
    tol = 1e-6
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -tol) or (x + r > 1 + tol) or (y - r < -tol) or (y + r > 1 + tol):
            message = (
                f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square"
            )
            return False, message
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(
                centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]
            )
            if dist + tol < radii[i] + radii[j]:
                message = f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}"
                return False, message
    return True, "success"


### <<< DEEPEVOLVE-BLOCK-END


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
    plt.savefig(f"circle_packing_{len(centers)}.png")


if __name__ == "__main__":
    for n in range(26, 33):
        centers, radii, _ = construct_packing(n=n)
        sum_radii = sum(radii)
        # print("radii", radii)

        valid_packing, message_packing = validate_packing(centers, radii)
        print(f"n={n}", "len(centers)", len(centers), "sum_radii", sum_radii, "valid_packing", valid_packing)
        # print("message_packing", message_packing)

        visualize(centers, radii)