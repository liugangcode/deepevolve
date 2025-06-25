# Circle Packing

## Problem Description

Given a positive integer *n*, the goal is to pack *n* disjoint circles inside a unit square in such a way as to maximize the sum of their radii. This problem focuses on discovering a new algorithm applicable to cases where *n* ranges from 26 to 32.

**Metric:** Sum of radii  
**Interface:** deepevolve_interface.py

### Mathematical Formulation

The objective is to maximize:

$$
\text{Objective} = \sum_{i=1}^{n} r_i
$$

subject to the following constraints:

- **Non-overlapping circles:**  
  For each pair of circles, the distance between their centers must be at least as large as the sum of their radii:
  
  ```math
  (x_i - x_j)^2 + (y_i - y_j)^2 \geq (r_i + r_j)^2 \quad \forall\, i \neq j
  ```

- **Boundary constraints:**  
  Each circle must lie entirely within the unit square:
  
  $$ 
  r_i \leq x_i \leq 1 - r_i \quad \text{and} \quad r_i \leq y_i \leq 1 - r_i \quad \forall\, i
  $$

Here, \(x_i\) and \(y_i\) represent the center coordinates of the \(i\)-th circle, and \(r_i\) its radius.

## Algorithmic Approach

The proposed method leverages `scipy.optimize.minimize` with the Sequential Least Squares Programming (SLSQP) algorithm. The problem is modeled as a constrained optimization task where both the center coordinates \((x_i, y_i)\) and the radius \(r_i\) of each circle are treated as decision variables.

Inequality constraints are formulated to:
- Prevent any pair of circles from overlapping.
- Ensure that all circles remain within the boundaries of the unit square.

Since SLSQP enforces constraints only within a numerical tolerance, it is important to note that the solution may occasionally permit slight violations (e.g., minor overlapping or circles slightly outside the unit square).

## Dependencies

- numpy
- scipy
- shapely

*Note:* No computational geometry libraries other than the ones listed above are to be used.

## Supplementary Material

For further details and insights on circle packing, please refer to the following resource:

[Circle Packing Supplementary Material](https://erich-friedman.github.io/packing/cirRsqu/)

> The initial idea is adapted from the output from [OpenEvolve](https://github.com/codelion/openevolve/tree/main/examples/circle_packing)