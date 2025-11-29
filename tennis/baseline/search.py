import argparse
import itertools
import sys
import os
import multiprocessing
import queue
import time
from types import SimpleNamespace

from tennis.baseline.train import train

# Global lock for printing to main console if needed (though we mostly log to files)
print_lock = multiprocessing.Lock()

def run_configuration(args_tuple):
    """
    Worker function to run a single configuration.
    args_tuple: (run_args_dict, device_queue, log_dir)
    """
    run_args_dict, device_queue, log_dir = args_tuple
    
    # Get a GPU device
    try:
        device = device_queue.get(timeout=60) # Wait for a device to become available
    except queue.Empty:
        with print_lock:
            print("Error: Could not acquire a GPU device.")
        return None, None, None

    # Update args with the assigned device
    run_args_dict["device"] = device
    args = SimpleNamespace(**run_args_dict)
    
    # Setup logging
    log_filename = f"search_h{args.hidden_size}_l{args.lstm_layers}_d{args.dropout}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Redirect stdout and stderr to the log file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    val_acc = None
    
    try:
        with open(log_path, "w") as log_file:
            sys.stdout = log_file
            sys.stderr = log_file
            
            print(f"Starting run on {device}")
            print(f"Config: hidden_size={args.hidden_size}, lstm_layers={args.lstm_layers}, dropout={args.dropout}")
            
            val_acc = train(args)
            
            print(f"Finished run. Val Acc: {val_acc}")
            
    except Exception as e:
        # If we can't write to log file, try to print to original stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        with print_lock:
            print(f"Run failed on {device} with error: {e}")
            import traceback
            traceback.print_exc()
    finally:
        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Return the device to the queue
        device_queue.put(device)

    return val_acc, run_args_dict, args.checkpoint_dir

def search():
    # Default arguments matching train.py
    defaults = {
        "feature_dir": "data/features",
        "checkpoint_dir": "tennis/baseline/models",
        "epochs": 300,
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "num_workers": 8,
        "device": None,
        # Hyperparams to search (defaults, will be overwritten)
        "hidden_size": 256,
        "lstm_layers": 1,
        "dropout": 0.1,
    }

    # Allow overriding defaults via command line for the search script itself
    parser = argparse.ArgumentParser(description="Hyperparameter Search for Tennis Classifier")
    parser.add_argument("--epochs", type=int, default=defaults["epochs"], help="Number of training epochs per run")
    parser.add_argument("--feature_dir", type=str, default=defaults["feature_dir"], help="Path to features")
    # device argument is removed/ignored as we manage devices internally
    
    script_args = parser.parse_args()

    # Create search_logs directory
    log_dir = "search_logs"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs will be saved to: {os.path.abspath(log_dir)}")

    # Search space
    hidden_sizes = [128, 256, 512]
    lstm_layers_list = [1, 2]
    dropouts = [0.1, 0.3, 0.5]

    combinations = list(itertools.product(hidden_sizes, lstm_layers_list, dropouts))
    print(f"Starting grid search with {len(combinations)} combinations...")

    # Setup multiprocessing
    # We assume 4 GPUs: cuda:0, cuda:1, cuda:2, cuda:3
    available_devices = [f"cuda:{i}" for i in range(4)]
    
    manager = multiprocessing.Manager()
    device_queue = manager.Queue()
    for d in available_devices:
        device_queue.put(d)

    # Prepare arguments for all runs
    tasks = []
    for hidden_size, lstm_layers, dropout in combinations:
        run_args_dict = defaults.copy()
        run_args_dict.update({
            "epochs": script_args.epochs,
            "feature_dir": script_args.feature_dir,
            "hidden_size": hidden_size,
            "lstm_layers": lstm_layers,
            "dropout": dropout,
            # Unique checkpoint dir for each run
            "checkpoint_dir": os.path.join(defaults["checkpoint_dir"], f"search_h{hidden_size}_l{lstm_layers}_d{dropout}")
        })
        tasks.append((run_args_dict, device_queue, log_dir))

    # Run with a pool of 4 workers (one for each GPU)
    # We use map_async or apply_async. Since we want to track results, map is fine.
    # However, since we are using a queue for devices, we need to make sure we don't deadlock.
    # With Pool(4) and 4 devices in queue, it should be fine.
    
    best_acc = 0.0
    best_params = {}
    best_checkpoint_dir = ""

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(run_configuration, tasks)

    print("\n" + "="*30)
    print("Search Complete")
    print("="*30)

    for val_acc, params, checkpoint_dir in results:
        if val_acc is not None:
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = {
                    "hidden_size": params["hidden_size"],
                    "lstm_layers": params["lstm_layers"],
                    "dropout": params["dropout"]
                }
                best_checkpoint_dir = checkpoint_dir

    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Best Model Location: {os.path.join(best_checkpoint_dir, 'best_model.pth')}")
    print("="*30)

if __name__ == "__main__":
    # Set start method to 'spawn' for CUDA compatibility if needed, 
    # though we are passing device strings, not tensors. 
    # 'spawn' is generally safer for PyTorch + multiprocessing.
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    search()
