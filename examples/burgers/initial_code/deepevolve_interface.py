import traceback
import warnings
from time import time
import numpy as np
import signal
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue

from main import main, Config
from solver import solver

def setup_signal_handler():
    """Setup signal handler to catch crashes"""
    def signal_handler(signum, frame):
        warnings.warn(f"Received signal {signum}. Process might be crashing.")
        return
    
    signal.signal(signal.SIGABRT, signal_handler)
    signal.signal(signal.SIGSEGV, signal_handler)

def cleanup_cuda_context():
    """Clean up CUDA context to prevent memory leaks"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        warnings.warn(f"Failed to cleanup CUDA context: {e}")

def run_single_nu_process(nu, timeout_sec, result_queue):
    """Run a single nu value in a separate process"""
    try:
        start_time = time()
        config = Config(nu=nu, base_dir="data_cache/burgers")
        metrics = main(solver, config)
        result = {
            "nu": nu,
            "nrmse": float(metrics["nrmse"]),
            "avg_rate": float(metrics["avg_rate"]),
            "time_in_minutes": (time() - start_time) / 60
        }
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", str(e)))

def run_single_nu_with_timeout(nu, timeout_sec=600):
    """Run a single nu value with its own timeout using multiprocessing"""
    result_queue = Queue()
    
    # Use multiprocessing instead of threading for better isolation
    process = Process(target=run_single_nu_process, args=(nu, timeout_sec, result_queue))
    process.start()
    
    # Wait for completion or timeout
    process.join(timeout_sec)
    
    if process.is_alive():
        warnings.warn(f"Nu={nu} runtime exceeded {timeout_sec/60:.2f} minutes. Terminating process.")
        process.terminate()
        process.join(5)  # Give it 5 seconds to terminate gracefully
        if process.is_alive():
            process.kill()  # Force kill if it doesn't terminate
        return None
    
    # Check if we got a result
    if not result_queue.empty():
        status, result = result_queue.get()
        if status == "success":
            return result
        else:
            warnings.warn(f"Error occurred for nu={nu}: {result}")
            return None
    else:
        warnings.warn(f"Nu={nu} did not return any result.")
        return None

def run_main_with_timeout(timeout_sec=1800):
    """Run main function with timeout and error handling"""
    setup_signal_handler()
    
    result = {"metrics": {}, "error": None}
    
    time_per_nu = timeout_sec // 3  # 600 seconds per nu
        
    try:
        # Initialize CUDA once at the beginning
        if torch.cuda.is_available():
            torch.cuda.init()
        
        for nu in [0.01, 0.1, 1.0]:
            try:
                cleanup_cuda_context()
                
                nu_result = run_single_nu_with_timeout(nu, time_per_nu)
                if nu_result is not None:
                    result["metrics"][nu] = nu_result
                    
            except Exception as e:
                warnings.warn(f"Critical error for nu={nu}: {str(e)}")
                cleanup_cuda_context()
                continue
                
    except Exception as e:
        result["error"] = str(e)
        warnings.warn(f"Global error in run_main_with_timeout: {str(e)}")
    finally:
        cleanup_cuda_context()
    
    return result["metrics"]

def deepevolve_interface():
    try:
        # Set multiprocessing start method at the beginning of the interface
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # If start method is already set, this will raise RuntimeError
            # This is fine, just continue
            pass
        
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start_time = time()
            results = run_main_with_timeout(1800)
            runtime = time() - start_time
            
        warning_messages = [str(w.message) for w in caught]
        
        metrics = {}
        combined_scores = []
        for nu in [0.01, 0.1, 1.0]:
            nu_metrics = results.get(nu, {
                "nrmse": None,
                "avg_rate": None,
                "time_in_minutes": None
            })
            if nu_metrics["nrmse"] is not None and nu_metrics["avg_rate"] is not None and nu_metrics["time_in_minutes"] is not None:
                current_combined_score = np.exp(nu_metrics["avg_rate"]) / (nu_metrics["nrmse"] + nu_metrics["time_in_minutes"])
                if np.isnan(current_combined_score):
                    current_combined_score = 0
            else:
                current_combined_score = 0
            combined_scores.append(current_combined_score)

            metrics[f'nu_{nu}_combined_score'] = current_combined_score
            metrics[f'nu_{nu}_nrmse'] = nu_metrics["nrmse"]
            metrics[f'nu_{nu}_convergence_rate'] = nu_metrics["avg_rate"]
            metrics[f'nu_{nu}_runtime_minutes'] = nu_metrics["time_in_minutes"]

        if warning_messages:
            warning_messages = list(set(warning_messages))
            if len(warning_messages) > 10:
                warning_messages = warning_messages[:10]
            metrics["program_warnings"] = warning_messages

        metrics["combined_score"] = float(np.mean(combined_scores))

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