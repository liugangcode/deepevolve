# Circle Packing

## Problem Description

Given a positive integer `n`, the goal is to pack `n` disjoint circles inside a unit square in a way that **maximizes the sum of their radii**. This task focuses on developing an algorithm that performs well for `n` ranging from 26 to 32. You may use the following libraries:
- `numpy`
- `scipy`
- `shapely`

Do **not** use any other computational geometry libraries.

- **Evaluation Metric**: Sum of the radii of the circles.
- **Interface File**: `deepevolve_interface.py`

---

## Initial Idea

We formulate the problem as a constrained optimization problem and use `scipy.optimize.minimize` with the **SLSQP** algorithm to find an arrangement of circles.

- Each circle's **center coordinates** and **radius** are treated as decision variables.
- **Inequality constraints** are added to ensure that no pair of circles overlaps.
- **Boundary constraints** are enforced so that all circles lie completely inside the unit square.

Note: The SLSQP optimizer respects inequalities only up to a numerical tolerance, so some outputs may include slightly overlapping circles or circles extending outside the square.

> The initial idea is adapted from the output from [OpenEvolve](https://github.com/codelion/openevolve/tree/main/examples/circle_packing)

For reference examples, see: [https://erich-friedman.github.io/packing/cirRsqu/](https://erich-friedman.github.io/packing/cirRsqu/)
