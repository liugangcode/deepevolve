import traceback
import warnings
from main import main_func
from time import time
import numpy as np


def deepevolve_interface():
    base_dir = "data_cache/amp_pd"
    # base_dir = "../../../data_cache/amp_pd"
    try:
        # Run main_func inside a warnings-catching context
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start_time = time()
            smape = main_func(base_dir)
            runtime = time() - start_time
            
        warning_messages = [str(w.message) for w in caught]

        # Compute combined score
        if np.isnan(smape):
            combined_score = 0.0
            print("smape is nan, set combined_score to 0.0")
        else:
            combined_score = 1 - smape / 200

        # Compute runtime in minutes, rounded
        runtime_minutes = round(runtime / 60, 2)

        # Compute improvement ratio
        initial_smape = 93.54330168877686
        ratio = (
            round((initial_smape - smape) / initial_smape * 100, 2)
            if not np.isnan(smape)
            else 0.0
        )

        # Build metrics dict
        metrics = {
            "combined_score": combined_score,
            "symmetric_mean_absolute_percentage_error (lower is better)": smape,
            "improvement_percentage_to_initial": ratio,
            "runtime_minutes": runtime_minutes,
        }
        if warning_messages:
            warning_messages = list(set(warning_messages))
            if len(warning_messages) > 10:
                warning_messages = warning_messages[:10]
            metrics["program_warnings"] = warning_messages

        return True, metrics

    except Exception as e:
        error_traceback = traceback.format_exc()
        error_info = (
            f"Error type: {type(e).__name__}\n"
            f"Error message: {e}\n"
            f"Traceback:\n{error_traceback}"
        )
        return False, error_info


if __name__ == "__main__":
    status, results = deepevolve_interface()
    print(f"Status: {status}")
    print(f"Results: {results}")
