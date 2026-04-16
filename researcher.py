import subprocess
import re
import json
import optuna
import os

# --- CONFIGURATION ---
TARGET_FILE = "train.py"
STATS_FILE = "bayesian_logs.json"
N_TRIALS = 10  # Number of Bayesian optimization experiments

def update_model_config(config):
    """Programmatically edit the train.py file with new hyperparameters."""
    with open(TARGET_FILE, "r") as f:
        lines = f.readlines()
    
    with open(TARGET_FILE, "w") as f:
        for line in lines:
            # Match parameters like: n_layer = 4
            for key, val in config.items():
                if line.strip().startswith(f"{key} ="):
                    # Cast val to int or float for clean file writing
                    val = round(val, 2) if isinstance(val, float) else val
                    line = f"{key} = {val}\n"
            f.write(line)

def execute_training():
    """Run the training script and capture the val_bpb output."""
    try:
        proc = subprocess.Popen(["python", TARGET_FILE], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate(timeout=120)
        
        # Regex to find: val_bpb: 1.234
        match = re.search(r"val_bpb: ([\d\.]+)", stdout)
        return float(match.group(1)) if match else None
    except Exception as e:
        print(f"Error: {e}")
        return None

def objective(trial):
    """The function Optuna will minimize."""
    # 1. Ask Optuna for the next hyperparameters to test
    config = {
        "d_model": trial.suggest_categorical("d_model", [128, 256, 384, 512]),
        "n_layer": trial.suggest_int("n_layer", 2, 8),
        "dropout": trial.suggest_float("dropout", 0.05, 0.25),
    }

    print(f"\n[TRIAL {trial.number}] TESTING CONFIG: {config}")
    
    # 2. Update config in train.py
    update_model_config(config)
    
    # 3. Run training and get result
    score = execute_training()
    
    if score is None:
        print("Trial failed to return a score.")
        return 10.0 # Return a penalty score if it fails
    
    print(f"Resulting BPB: {score}")
    return score

def main():
    print("-" * 40)
    print("BAYESIAN AUTO-RESEARCHER: BOOTING...")
    print("-" * 40)
    
    # Create the Optuna Study (minimizing loss/bpb)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    # Save Results
    results = {
        "best_trial": {
            "number": study.best_trial.number,
            "params": study.best_params,
            "value": study.best_value
        },
        "all_trials": [
            {"number": t.number, "params": t.params, "value": t.value} for t in study.trials
        ]
    }
    
    with open(STATS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 40)
    print(f"BAYESIAN RESEARCH COMPLETE!")
    print(f"Best Params: {study.best_params}")
    print(f"Best BPB: {study.best_value}")
    print("=" * 40)

if __name__ == "__main__":
    main()
