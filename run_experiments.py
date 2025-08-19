import os
import subprocess
import time
import numpy as np

# --- Experiment Configuration ---
base_output_dir = f"ga_runs/experiments_{int(time.time())}"
thresholds_to_test = np.arange(0.1, 4.1, 0.1) # 0.1, 0.2, ..., 4.0

# --- Launch Experiments ---
processes = []
for threshold in thresholds_to_test:
    threshold_str = f"{threshold:.1f}"
    experiment_dir = os.path.join(base_output_dir, f"threshold_{threshold_str}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    log_file_path = os.path.join(experiment_dir, "training.log")
    
    command = [
        "python3",
        "run_experiment.py",
        "--loss_threshold",
        str(threshold),
        "--output_dir",
        experiment_dir
    ]
    
    print(f"Starting experiment for threshold {threshold_str}...")
    print(f"  -> Log file: {log_file_path}")
    
    with open(log_file_path, 'w') as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((process, threshold_str))

print(f"\nLaunched {len(processes)} experiments. You can monitor their progress in the '{base_output_dir}' directory.")

# --- Wait for all processes to complete ---
for process, threshold_str in processes:
    process.wait()
    print(f"Experiment for threshold {threshold_str} has completed.")

print("\nAll experiments have finished.")
