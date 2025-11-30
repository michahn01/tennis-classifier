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

# Global lock for printing to main console if needed (though we mostly log to files)
print_lock = multiprocessing.Lock()


def worker(device_queue, task_queue, result_queue, log_dir):
    """
    Worker process that picks up tasks and runs them on an available device.
    """
    while True:
        try:
            # Get a task (non-blocking, or with timeout if we want to be safe)
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
        "--epochs", type=int, default=300, help="Number of training epochs per run"
    )
    parser.add_argument(
        "--feature_dir", type=str, default="data/features", help="Path to features"
    )
    parser.add_argument(
        "--k_folds", type=int, default=0, help="Number of folds for CV (0 to disable)"
    )
    parser.add_argument(
        "--pose_dir",
        type=str,
        default="data/pose_features",
        help="Path to pre-extracted pose features (required for skeleton model)",
    )
    # device argument is removed/ignored as we manage devices internally

    script_args = parser.parse_args()

    # Default arguments matching train.py
    defaults = {
        "model_type": script_args.model_type,
        "feature_dir": script_args.feature_dir,
        "pose_dir": script_args.pose_dir,
        "checkpoint_dir": f"tennis/{script_args.model_type}/models",
        "epochs": 300,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "num_workers": 4,
        "device": None,
        # Hyperparams to search (defaults, will be overwritten)
        "hidden_size": 256,
        "lstm_layers": 1,
        "dropout": 0.1,
    }

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

    # Setup multiprocessing queues
    manager = multiprocessing.Manager()
    device_queue = manager.Queue()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    # We assume 4 GPUs: cuda:0, cuda:1, cuda:2, cuda:3
    available_devices = [f"cuda:{i}" for i in range(4)]
    tasks_per_gpu = 4
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
    # We start 4 * tasks_per_gpu workers.
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
        target_path = os.path.join(target_dir, "total_best_model.pth")
        shutil.copy2(best_model_src, target_path)
        print(f"Best model saved to: {target_path}")
        print(f"(Source: {best_model_src})")
    else:
        print("Error: Could not locate best model file.")
    print("=" * 30)


if __name__ == "__main__":
    # Set start method to 'spawn' for CUDA compatibility if needed,
    # though we are passing device strings, not tensors.
    # 'spawn' is generally safer for PyTorch + multiprocessing.
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    search()
