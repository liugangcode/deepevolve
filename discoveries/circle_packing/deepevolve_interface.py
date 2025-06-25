from main import construct_packing, validate_packing
from time import time
import numpy as np
import traceback
import warnings  # DEBUG: imported warnings for adaptive_bisection in main.py
import warnings
import signal
from contextlib import contextmanager


@contextmanager
def timeout(duration):
    """Context manager for timing out function calls"""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function call timed out after {duration} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)

    try:
        yield
    finally:
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def deepevolve_interface():
    try:
        start_time = time()

        # SOTA values for comparison
        sota_values = {
            26: 2.6358627564136983,
            27: 2.685,
            28: 2.737,
            29: 2.790,
            30: 2.842,
            31: 2.889,
            32: 2.937944526205518,
        }

        all_results = {}
        all_sum_radii = []

        # Run for n from 26 to 32
        for n in range(26, 33):
            # Apply 1-minute timeout to construct_packing
            try:
                with timeout(60):
                    centers, radii, sum_radii = construct_packing(n=n)

                if not isinstance(centers, np.ndarray):
                    centers = np.array(centers)
                if not isinstance(radii, np.ndarray):
                    radii = np.array(radii)

                # Validate solution
                valid_packing, message_packing = validate_packing(centers, radii)

                if not valid_packing:
                    print(f"Invalid packing for n={n}: {message_packing}")

            except TimeoutError as te:
                warnings.warn(
                    f"Timeout occurred for n={n}: {te}. Setting sum_radii to 0."
                )
                centers = np.array([])
                radii = np.array([])
                sum_radii = 0.0
                valid_packing = False
                message_packing = f"60s Timeout occurred for n={n}"

            # Store results
            all_results[n] = {
                "sum_radii": sum_radii if valid_packing else 0.0,
                "valid": valid_packing,
                "message": message_packing,
            }
            all_sum_radii.append(sum_radii if valid_packing else 0.0)

        # Calculate runtime in seconds
        runtime = time() - start_time
        runtime = round(runtime, 2)

        combined_score = np.mean(all_sum_radii)

        metrics = {
            "combined_score": combined_score,
            "runtime_seconds": runtime,
        }

        # Add individual sum_radii and ratios to SOTA for each n
        for n in range(26, 33):
            result = all_results[n]
            sum_radii = result["sum_radii"]
            valid = result["valid"]

            # Add sum_radii for this n
            metrics[f"sum_radii_for_n_{n}"] = sum_radii

            # Calculate ratio to SOTA
            if n in sota_values and valid:
                sota_value = sota_values[n]
                ratio_to_sota = sum_radii / sota_value
                metrics[f"ratio_to_sota_for_n_{n}"] = ratio_to_sota
            else:
                metrics[f"ratio_to_sota_for_n_{n}"] = 0.0

            # Add validity for this n
            metrics[f"validity_for_n_{n}"] = 1.0 if valid else 0.0
            if not valid:
                metrics[f"message_for_n_{n}"] = message_packing

        overall_validity = all(all_results[n]["valid"] for n in range(26, 33))
        metrics["overall_validity"] = 1.0 if overall_validity else 0.0

        return True, metrics

    except Exception as e:
        # Capture full traceback information
        error_traceback = traceback.format_exc()
        error_info = f"""
            Error type: {type(e).__name__}
            Error message: {str(e)}
            Traceback: {error_traceback}
        """
        return False, error_info


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
    # plt.savefig('circle_packing.png')


if __name__ == "__main__":
    status, metrics = deepevolve_interface()
    print(f"Status: {status}")
    print(f"Metrics: {metrics}")
    # AlphaEvolve improved this to 2.635


