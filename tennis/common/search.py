import argparse
import itertools
import sys
import os
import multiprocessing
import queue
import shutil
from types import SimpleNamespace

from tennis.common.train import train
from tennis.common.model_utils import get_model_choices
import tennis.common.config as config

# Global lock for printing to main console if needed (though we mostly log to files)
print_lock = multiprocessing.Lock()


def worker(device_queue, task_queue, result_queue, log_dir):
    """Worker process that picks up tasks and runs them on an available device."""
    while True:
        try:
            # Get a task (non-blocking, or with timeout to be safe)
            task = task_queue.get(timeout=1)
        except queue.Empty:
            break

        run_args_dict, config_name = task

        # Get a GPU device
        try:
            device = device_queue.get(timeout=60)
        except queue.Empty:
            with print_lock:
                print("Error: Could not acquire a GPU device.")
            return

        # Update args with the assigned device
        run_args_dict["device"] = device
        args = SimpleNamespace(**run_args_dict)

        # Setup logging
        log_filename = f"{config_name}.log"
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
                print(f"Config: {config_name}")

                val_acc = train(args)

                print(f"Finished run. Val Acc: {val_acc}")

        except Exception as e:
            # If we can't write to log file, try to print to original stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            with print_lock:
                print(f"Run {config_name} failed on {device} with error: {e}")
                import traceback

                traceback.print_exc()
        finally:
            # Restore stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            # Return the device to the queue
            device_queue.put(device)

        result_queue.put((val_acc, run_args_dict, args.checkpoint_dir))


def search():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Search for Tennis Classifier"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=get_model_choices(),
        help="Type of model to search for",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.DEFAULT_EPOCHS,
        help="Number of training epochs per run",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=config.DEFAULT_FEATURE_DIR,
        help="Path to features",
    )
    parser.add_argument(
        "--k_folds", type=int, default=0, help="Number of folds for CV (0 to disable)"
    )
    parser.add_argument(
        "--pose_dir",
        type=str,
        default=config.DEFAULT_POSE_DIR,
        help="Path to pre-extracted pose features (required for skeleton model)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results and logs (default: results/{model_type}/)",
    )
    # device argument is removed/ignored as we manage devices internally

    script_args = parser.parse_args()

    # Default arguments matching train.py
    defaults = {
        "model_type": script_args.model_type,
        "feature_dir": script_args.feature_dir,
        "pose_dir": script_args.pose_dir,
        "checkpoint_dir": f"tennis/{script_args.model_type}/models",
        "epochs": config.DEFAULT_EPOCHS,
        "batch_size": config.DEFAULT_BATCH_SIZE,
        "lr": config.SEARCH_LR,
        "weight_decay": config.DEFAULT_WEIGHT_DECAY,
        "num_workers": config.SEARCH_NUM_WORKERS,
        "device": None,
        # Hyperparams to search (defaults, will be overwritten)
        "hidden_size": config.DEFAULT_HIDDEN_SIZE,
        "lstm_layers": config.DEFAULT_LSTM_LAYERS,
        "dropout": config.DEFAULT_DROPOUT,
    }

    # Create output directory
    if script_args.output_dir:
        output_dir = script_args.output_dir
    else:
        output_dir = f"results/{script_args.model_type}/"

    if os.path.exists(output_dir):
        print(f"Clearing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")

    # Setup main process logging
    log_file_path = os.path.join(output_dir, "search.log")
    # We will tee the output to both stdout and the log file
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush() # Ensure immediate write
        def flush(self):
            for f in self.files:
                f.flush()

    f_log = open(log_file_path, 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f_log)
    
    # Worker logs directory
    log_dir = os.path.join(output_dir, "worker_logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Worker logs will be saved to: {os.path.abspath(log_dir)}")

    # Search space
    hidden_sizes = config.SEARCH_SPACE["hidden_sizes"]
    lstm_layers_list = config.SEARCH_SPACE["lstm_layers"]
    dropouts = config.SEARCH_SPACE["dropouts"]

    combinations = list(itertools.product(hidden_sizes, lstm_layers_list, dropouts))
    print(f"Starting grid search with {len(combinations)} combinations...")

    # Setup multiprocessing queues
    manager = multiprocessing.Manager()
    device_queue = manager.Queue()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    # We assume 4 GPUs: cuda:0, cuda:1, cuda:2, cuda:3
    available_devices = [f"cuda:{i}" for i in range(4)]
    tasks_per_gpu = config.SEARCH_TASKS_PER_GPU
    for d in available_devices:
        for _ in range(tasks_per_gpu):
            device_queue.put(d)

    # Prepare tasks
    for hidden_size, lstm_layers, dropout in combinations:
        base_config_name = f"search_h{hidden_size}_l{lstm_layers}_d{dropout}"

        if script_args.k_folds > 0:
            for fold_idx in range(script_args.k_folds):
                run_args_dict = defaults.copy()
                config_name = f"{base_config_name}_fold{fold_idx}"

                run_args_dict.update(
                    {
                        "epochs": script_args.epochs,
                        "feature_dir": script_args.feature_dir,
                        "hidden_size": hidden_size,
                        "lstm_layers": lstm_layers,
                        "dropout": dropout,
                        "k_folds": script_args.k_folds,
                        "fold_index": fold_idx,
                        # Unique checkpoint dir for each run
                        "checkpoint_dir": os.path.join(
                            defaults["checkpoint_dir"],
                            base_config_name,
                            f"fold{fold_idx}",
                        ),
                        # Directory for learning curve graphs
                        "plot_dir": os.path.join(log_dir, f"{config_name}_graphs"),
                    }
                )
                task_queue.put((run_args_dict, config_name))
        else:
            run_args_dict = defaults.copy()
            config_name = base_config_name
            run_args_dict.update(
                {
                    "epochs": script_args.epochs,
                    "feature_dir": script_args.feature_dir,
                    "hidden_size": hidden_size,
                    "lstm_layers": lstm_layers,
                    "dropout": dropout,
                    "k_folds": 0,
                    # Unique checkpoint dir for each run
                    "checkpoint_dir": os.path.join(
                        defaults["checkpoint_dir"], config_name
                    ),
                    # Directory for learning curve graphs
                    "plot_dir": os.path.join(log_dir, f"{config_name}_graphs"),
                }
            )
            task_queue.put((run_args_dict, config_name))

    # Start worker processes
    # We start len(available_devices)=4 * tasks_per_gpu workers.
    # They will pull from task_queue until empty.
    processes = []
    for _ in range(len(available_devices) * tasks_per_gpu):
        p = multiprocessing.Process(
            target=worker, args=(device_queue, task_queue, result_queue, log_dir)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("\n" + "=" * 30)
    print("Search Complete")
    print("=" * 30)

    # Collect results
    best_acc = 0.0
    best_params = {}
    best_model_src = ""

    results_by_config = {}

    while not result_queue.empty():
        val_acc, params, checkpoint_dir = result_queue.get()
        if val_acc is not None:
            # Reconstruct base config name from params
            h = params["hidden_size"]
            l_num = params["lstm_layers"]
            d = params["dropout"]
            config_key = f"h{h}_l{l_num}_d{d}"

            if config_key not in results_by_config:
                results_by_config[config_key] = {
                    "accs": [],
                    "params": params,
                    "dirs": [],
                }

            results_by_config[config_key]["accs"].append(val_acc)
            results_by_config[config_key]["dirs"].append(checkpoint_dir)

    for config_key, data in results_by_config.items():
        avg_acc = sum(data["accs"]) / len(data["accs"])
        print(
            f"Config {config_key}: Avg Acc = {avg_acc:.2f}% (Runs: {len(data['accs'])})"
        )

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_params = data["params"]

            # Find the fold with the highest accuracy in this config
            best_fold_idx = data["accs"].index(max(data["accs"]))
            best_fold_dir = data["dirs"][best_fold_idx]
            best_model_src = os.path.join(best_fold_dir, "best_model.pth")

    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    if best_model_src and os.path.exists(best_model_src):
        target_dir = f"tennis/{script_args.model_type}/models"
        os.makedirs(target_dir, exist_ok=True)
        
        # Construct filename with hyperparams
        h = best_params["hidden_size"]
        l_num = best_params["lstm_layers"]
        d = best_params["dropout"]
        target_filename = f"best_model_h{h}_l{l_num}_d{d}.pth"
        
        target_path = os.path.join(target_dir, target_filename)
        shutil.copy2(best_model_src, target_path)
        print(f"Best model saved to: {target_path}")
        print(f"(Source: {best_model_src})")
    else:
        print("Error: Could not locate best model file.")

    
    # Run test on best model
    if best_model_src and os.path.exists(best_model_src):
        print("\n" + "=" * 30)
        print("Running Test on Best Model")
        print("=" * 30)
        
        from tennis.common.test import test
        
        # Construct args for test
        test_args = SimpleNamespace(
            model_type=script_args.model_type,
            feature_dir=script_args.feature_dir,
            pose_dir=script_args.pose_dir,
            model_path=best_model_src, # Use the source file directly or the copied one? Let's use source to be safe/direct.
            # Actually, let's use the copied one if we want to verify that too, but source is fine.
            # Wait, the prompt says "test the model on the file of the best model found".
            # Let's use the one we just identified.
            
            # Hyperparams
            hidden_size=best_params["hidden_size"],
            lstm_layers=best_params["lstm_layers"],
            dropout=best_params["dropout"],
            
            # Other defaults
            batch_size=config.DEFAULT_BATCH_SIZE,
            num_workers=config.DEFAULT_NUM_WORKERS,
            device=None, # Let test.py decide or pass one if we want
            infer_params=False,
            save_dir=output_dir # Save confusion matrix etc to the main output dir
        )
        
        # Capture test output to log file
        test_log_path = os.path.join(output_dir, "test_results.log")
        
        # We need to capture the output of the test function. 
        # Since we are already teeing stdout to search.log, we might want to separate this.
        # But the requirement says "The text output/results of this test should go into a separate log file".
        # So we should temporarily redirect stdout to ONLY the test log file.
        
        try:
            with open(test_log_path, "w") as test_log_file:
                # Temporarily redirect stdout to the test log file
                # We also want to see it in the main log/console? 
                # "The text output/results of this test should go into a separate log file"
                # It doesn't explicitly say it shouldn't be in the main log, but usually "separate" implies distinct.
                # However, for debugging it's nice to see. 
                # Let's just redirect to the file as requested.
                
                sys.stdout = test_log_file
                test(test_args)
        except Exception as e:
             sys.stdout = original_stdout # Restore first
             # We are still wrapped in the Tee from earlier, so this restores to (console + search.log)
             # Wait, original_stdout was the ACTUAL console. 
             # We want to restore to the Tee object if we want to continue logging to search.log?
             # Actually, we are at the end of the script.
             
             # Let's restore to the Tee object
             sys.stdout = Tee(original_stdout, f_log)
             print(f"Error running test: {e}")
             import traceback
             traceback.print_exc()
        finally:
             # Restore stdout to the Tee object (or original if we are done)
             sys.stdout = Tee(original_stdout, f_log)

    print("=" * 30)
    
    # Close the search log file
    sys.stdout = original_stdout
    f_log.close()


if __name__ == "__main__":
    # Set start method to 'spawn' for CUDA compatibility if needed,
    # though we are passing device strings, not tensors.
    # 'spawn' is generally safer for PyTorch + multiprocessing.
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    search()
