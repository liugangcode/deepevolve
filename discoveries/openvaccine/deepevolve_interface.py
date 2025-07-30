import traceback
import warnings
from main import main
from time import time
import numpy as np
import multiprocessing


def run_main_with_timeout(base_dir, timeout_sec):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    def target():
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                metrics = main(base_dir)

            warning_messages = [str(w.message) for w in caught]
            return_dict["metrics"] = metrics
            if len(warning_messages) > 10:
                warning_messages = warning_messages[:10]
            return_dict["warnings"] = warning_messages
            return_dict["error"] = None
        except Exception as e:
            return_dict["metrics"] = None
            return_dict["warnings"] = []
            return_dict["error"] = str(e)

    p = multiprocessing.Process(target=target)
    p.start()
    p.join(timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(
            f"The model runtime exceeded {timeout_sec/60:.2f} minutes and was terminated. Please reduce the runtime of the model."
        )

    if return_dict["error"]:
        raise Exception(return_dict["error"])

    return return_dict["metrics"], return_dict.get("warnings", [])


def deepevolve_interface():
    base_dir = "data_cache/openvaccine"
    # base_dir = "../../../data_cache/openvaccine"
    try:
        start_time = time()
        metrics, subprocess_warnings = run_main_with_timeout(base_dir, 1800)
        runtime = time() - start_time

        runtime_minutes = round(runtime / 60, 2)

        test_score = metrics["test_MCRMSE"]
        if np.isnan(test_score):
            test_score = 999

        initial_score = 0.3914539605379105
        first_place_score = 0.34198
        improvement_to_initial = round(
            (initial_score - test_score) / initial_score * 100, 2
        )
        improvement_to_first_place = round(
            (first_place_score - test_score) / first_place_score * 100, 2
        )

        metrics = {
            "combined_score": 1 / (1 + test_score),
            "improvement_percentage_to_initial": improvement_to_initial,
            "improvement_percentage_to_first_place": improvement_to_first_place,
            "runtime_minutes": runtime_minutes,
            "test_MCRMSE_lower_is_better": test_score,
            "train_mean_loss_across_folds_lower_is_better": metrics[
                "train_mean_loss_across_folds"
            ],
        }

        # Include warnings from subprocess
        if subprocess_warnings:
            warning_messages = list(set(subprocess_warnings))
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
