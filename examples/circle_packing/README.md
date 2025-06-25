# Circle Packing

## Problem Description

The Circle Packing problem involves arranging a given number of disjoint circles, denoted by the positive integer \( n \), within a unit square. The objective is to maximize the sum of their radii. This problem is particularly focused on developing a novel algorithm applicable for \( n \) ranging from 26 to 32. The solution must adhere to the following constraints and requirements:

- **Constraints**:
  - Circles must be disjoint, meaning no two circles should overlap.
  - All circles must be contained within the boundaries of the unit square.

- **Libraries and Requirements**:
  - You are allowed to use the following libraries: `numpy`, `scipy`, and `shapely`.
  - Usage of any other computational geometry libraries is prohibited.

- **Evaluation Metric**: The objective is to maximize the **sum of radii** of the circles.

- **Interface File**: The implementation should be compatible with the interface provided in `deepevolve_interface.py`.

## Initial Idea

The initial approach to solving the Circle Packing problem employs the `scipy.optimize.minimize` function with the Sequential Least Squares Programming (SLSQP) algorithm. This method is used to identify the optimal arrangement of circles within the unit square. The problem is formulated as a constrained optimization problem where both the center coordinates and the radii of the circles are treated as decision variables.

Key aspects of the approach include:

- **Inequality Constraints**: These are added to ensure that no two circles overlap. The constraints are designed to maintain a minimum distance between the centers of any two circles, which must be at least the sum of their radii.

- **Boundary Constraints**: These ensure that all circles remain within the confines of the unit square. This involves setting constraints on the center coordinates and radii to prevent any part of a circle from extending beyond the square's edges.

- **Numerical Tolerance**: The SLSQP algorithm attempts to satisfy all constraints, but only within a numerical tolerance. This may occasionally result in solutions where circles overlap slightly or extend marginally outside the unit square.

For further insights and related research, you can refer to [this supplement](https://erich-friedman.github.io/packing/cirRsqu/).