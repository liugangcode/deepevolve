import traceback
import warnings
from main import main
from time import time
import numpy as np
import multiprocessing


def run_main_with_timeout(base_dir, timeout_sec):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    ### >>> DEEPEVOLVE-BLOCK-START: Capture full traceback in exception handling within target function
    def target():
        try:
            return_dict["metrics"] = main(base_dir)
            return_dict["error"] = None
        except Exception as e:
            import traceback

            return_dict["metrics"] = None
            return_dict["error"] = traceback.format_exc()

    ### <<< DEEPEVOLVE-BLOCK-END

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

    return return_dict["metrics"]


def deepevolve_interface():
    base_dir = "data_cache/usp_p2p"
    # base_dir = "../../../data_cache/usp_p2p"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start_time = time()
            metrics = run_main_with_timeout(base_dir, 1800)
            # metrics = main(base_dir)
            runtime = time() - start_time

        warning_messages = [str(w.message) for w in caught]
        runtime_minutes = round(runtime / 60, 2)

        initial_score = 0.803648329426078
        ratio = round(
            (metrics["eval_pearson"] - initial_score) / initial_score * 100, 2
        )

        metrics = {
            "combined_score": metrics["eval_pearson"],
            "improvement_percentage_to_initial": ratio,
            "runtime_minutes": runtime_minutes,
            "eval_loss": metrics["eval_loss"],
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
