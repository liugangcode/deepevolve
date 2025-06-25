# circle_packing

## Problem Description

Given a positive integer \( n \), the goal is to pack \( n \) disjoint circles inside a unit square in such a way that the sum of their radii is maximized. The problem specifically targets values of \( n \) ranging from 26 to 32. The solution should utilize only the following libraries:

- numpy
- scipy
- shapely

**Constraints and Requirements:**

- Each circle must be fully contained within a unit square.
- Circles must not overlap.
- The objective is to maximize the sum of the circle radii, given by:
  \[
  \text{maximize} \quad \sum_{i=1}^{n} r_i
  \]

- **Evaluation Metric**: - **Evaluation Metric**: sum of radii
- **Interface File**: - **Interface File**: `deepevolve_interface.py`

## Initial Idea

The proposed approach leverages the SLSQP algorithm provided by `scipy.optimize.minimize`. The idea is to reformulate circle packing as a constrained optimization problem where both the center coordinates and the radius of each circle are treated as decision variables. The steps include:

- **Formulation**: Cast the problem into a constrained optimization framework.
- **Constraints**:
  - **Inequality Constraints**: To ensure no pair of circles overlaps, include constraints that maintain a minimum distance between circle centers.
  - **Boundary Constraints**: Impose constraints that require every circle to be contained within the confines of the unit square.
- **Optimization Algorithm**: Utilize the SLSQP (Sequential Least SQuares Programming) algorithm which will aim to satisfy each inequality within a specified numerical tolerance. This may occasionally result in slight violations such as overlapping circles or circles extending outside the unit square due to approximate enforcement of the constraints.

For further reference, please consult this [supplementary resource](https://erich-friedman.github.io/packing/cirRsqu/).