import traceback
from main import main, Config
from time import time
import warnings
import threading
import signal

def run_main_with_timeout(config, timeout_sec):
    result = {"metrics": None, "error": None}
    
    def target():
        try:
            result["metrics"] = main(config)
        except Exception as e:
            result["error"] = str(e)
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_sec)
    
    if thread.is_alive():
        # Note: Cannot forcefully kill a thread in Python, but the daemon thread will exit when main exits
        raise TimeoutError(f"The model runtime exceeded {timeout_sec/60:.2f} minutes and was terminated. Please reduce the runtime of the model.")
    
    if result["error"]:
        raise Exception(result["error"])
    
    return result["metrics"]

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