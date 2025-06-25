import traceback
from main_pyg import config_and_run
from utils import get_args
from time import time
import warnings


def deepevolve_interface():
    args = get_args()
    # args.base_dir = "../../../data_cache/polymer"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start_time = time()
            results, wmae, r2 = config_and_run(args)
            runtime = time() - start_time

        warning_messages = [str(w.message) for w in caught]

        runtime = round(runtime / 60, 2)

        current_combined_score = 1 / (1 + wmae) * 0.5 + r2 * 0.5
        metrics = {
            "combined_score": current_combined_score,
            "wmae_inverse": 1 / (1 + wmae),
            "r2_avg": r2,
            "runtime_minutes": runtime,
            **results,
        }
        if warning_messages:
            warning_messages = list(set(warning_messages))
            if len(warning_messages) > 10:
                warning_messages = warning_messages[:10]
            metrics["program_warnings"] = warning_messages

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


if __name__ == "__main__":
    status, results = deepevolve_interface()
    print(f"Status: {status}")
    print(f"Results: {results}")
