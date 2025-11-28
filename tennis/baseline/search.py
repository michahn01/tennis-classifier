import argparse
import itertools
import sys
import os
from types import SimpleNamespace


from tennis.baseline.train import train

def search():
    # Default arguments matching train.py
    defaults = {
        "feature_dir": "data/features",
        "checkpoint_dir": "tennis/baseline/models",
        "epochs": 300,
        "batch_size": 8,
        "lr": 1e-4,
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
    parser.add_argument("--device", type=str, default=defaults["device"], help="Device to use")
    
    script_args = parser.parse_args()

    # Search space
    hidden_sizes = [128, 256, 512]
    lstm_layers_list = [1, 2]
    dropouts = [0.1, 0.3, 0.5]

    best_acc = 0.0
    best_params = {}

    combinations = list(itertools.product(hidden_sizes, lstm_layers_list, dropouts))
    print(f"Starting grid search with {len(combinations)} combinations...")

    for i, (hidden_size, lstm_layers, dropout) in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing: hidden_size={hidden_size}, lstm_layers={lstm_layers}, dropout={dropout}")
        
        # Create args for this run
        run_args_dict = defaults.copy()
        run_args_dict.update({
            "epochs": script_args.epochs,
            "feature_dir": script_args.feature_dir,
            "device": script_args.device,
            "hidden_size": hidden_size,
            "lstm_layers": lstm_layers,
            "dropout": dropout,
            # Unique checkpoint dir for each run to avoid conflicts/overwrites during search
            "checkpoint_dir": os.path.join(defaults["checkpoint_dir"], f"search_h{hidden_size}_l{lstm_layers}_d{dropout}")
        })
        
        args = SimpleNamespace(**run_args_dict)
        
        try:
            val_acc = train(args)
            
            if val_acc is None:
                print("Training failed or no data found.")
                continue

            print(f"Result: Val Acc = {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = {
                    "hidden_size": hidden_size,
                    "lstm_layers": lstm_layers,
                    "dropout": dropout
                }
                print(f"New best found! {best_params} with Acc: {best_acc:.2f}%")
                
        except Exception as e:
            print(f"Run failed with error: {e}")

    print("\n" + "="*30)
    print("Search Complete")
    print("="*30)
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print("Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("="*30)

if __name__ == "__main__":
    search()
