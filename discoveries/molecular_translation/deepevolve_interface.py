import traceback
import warnings
from time import time
import threading

from main import main, Config


def run_main_with_timeout(base_dir, timeout_sec):
    result = {"metrics": None, "error": None}

    def target():
        try:
            result["metrics"] = main(Config(base_dir=base_dir))
        except Exception as e:
            result["error"] = str(e)

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_sec)

    if thread.is_alive():
        raise TimeoutError(
            f"The model runtime exceeded {timeout_sec/60:.2f} minutes and was terminated. Please reduce the runtime of the model."
        )

    if result["error"]:
        raise Exception(result["error"])

    return result["metrics"]


def deepevolve_interface():
    # base_dir = "../../../data_cache/molecular_translation"
    base_dir = "data_cache/molecular_translation"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start_time = time()
            results = run_main_with_timeout(base_dir, 1800)
            runtime = time() - start_time

        warning_messages = [str(w.message) for w in caught]
        runtime_minutes = round(runtime / 60, 2)

        scores = 1 - float(results)

        metrics = {
            "combined_score": scores,
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

