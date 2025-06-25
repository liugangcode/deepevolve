import traceback
from main_pyg import config_and_run
from utils import get_args
from time import time
import warnings


def deepevolve_interface():
    args = get_args()
    args.dataset = "ogbg-molsider"
    args.by_default = True
    args.trials = 3

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start_time = time()
            results = config_and_run(args)
            runtime = time() - start_time

        warning_messages = [str(w.message) for w in caught]

        runtime = round(runtime / 60, 2)
        auc_mean = results["test_auc_mean"]
        auc_std = results["test_auc_std"]
        initial_combined_score = 0.7914562889678236
        current_combined_score = auc_mean * 0.5 + (1 - auc_std) * 0.5
        impr_pct = (
            (current_combined_score - initial_combined_score)
            / initial_combined_score
            * 100
        )
        metrics = {
            "combined_score": current_combined_score,
            "improvement_percentage_to_initial": impr_pct,
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


