import traceback
from main import main, Config
from time import time
import warnings
import threading
import signal


# DEBUG: module-level worker function for spawn pickling compatibility
### >>> DEEPEVOLVE-BLOCK-START: Enhance error reporting in _worker_main
def _worker_main(cfg, q):
    try:
        metrics = main(cfg)
        q.put(("metrics", metrics))
    except Exception as e:
        import traceback

        q.put(("error", traceback.format_exc()))


### <<< DEEPEVOLVE-BLOCK-END


def run_main_with_timeout(config, timeout_sec):
    # DEBUG: Use a separate process instead of thread to safely run GPU operations and allow termination
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    # DEBUG: use module-level worker function for spawn pickling compatibility
    process = ctx.Process(target=_worker_main, args=(config, queue))
    # DEBUG: Using 'spawn' start method via multiprocessing context to avoid CUDA reinitialization in forked subprocess
    process.start()
    process.join(timeout_sec)

    if process.is_alive():
        process.terminate()
        raise TimeoutError(
            f"The model runtime exceeded {timeout_sec/60:.2f} minutes and was terminated. Please reduce the runtime of the model."
        )

    if not queue.empty():
        key, value = queue.get()
        if key == "error":
            raise Exception(value)
        return value
    else:
        raise Exception(
            "No result returned from the model run within the allotted time."
        )


def deepevolve_interface():
    config = Config()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start_time = time()
            # results = main(config)
            results = run_main_with_timeout(config, 1800)
            runtime = time() - start_time

        warning_messages = [str(w.message) for w in caught]

        runtime = round(runtime / 60, 2)

        train_map = results["train_map"]
        valid_map = results["valid_map"]
        test_map = results["test_map"]

        metrics = {
            "combined_score": test_map,
            "train_map": train_map,
            "valid_map": valid_map,
            "test_map": test_map,
            "runtime_minutes": runtime,
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
