import os
import re
import pandas as pd

def parse_log_file(file_path):
    """Parses a log file to extract key metrics."""
    with open(file_path, 'r') as f:
        content = f.read()

    final_mask_weight_match = re.findall(r"Mask Weight: (\d\.\d+)", content)
    all_manhattan_match = re.findall(r"Validation Set Manhattan Distance: (\d+\.\d+)", content)
    
    final_mask_weight = float(final_mask_weight_match[-1]) if final_mask_weight_match else 1.0
    
    if all_manhattan_match:
        best_manhattan = min([float(m) for m in all_manhattan_match])
    else:
        best_manhattan = None

    return {
        "final_mask_weight": final_mask_weight,
        "best_manhattan_distance": best_manhattan,
        "reached_zero_mask": final_mask_weight is not None and final_mask_weight < 1e-6
    }

def analyze_experiments(base_dir):
    """Analyzes all experiment subdirectories in a base directory."""
    results = []
    for dir_name in os.listdir(base_dir):
        if dir_name.startswith("threshold_"):
            experiment_dir = os.path.join(base_dir, dir_name)
            log_file = os.path.join(experiment_dir, "training.log")
            
            if os.path.exists(log_file):
                threshold = float(dir_name.replace("threshold_", ""))
                metrics = parse_log_file(log_file)
                metrics["threshold"] = threshold
                results.append(metrics)

    return pd.DataFrame(results)

if __name__ == '__main__':
    # Find the latest experiment directory
    all_experiment_dirs = [d for d in os.listdir('ga_runs') if d.startswith('experiments_')]
    if not all_experiment_dirs:
        print("No experiment directories found in 'ga_runs/'.")
    else:
        latest_dir = sorted(all_experiment_dirs)[-1]
        latest_path = os.path.join('ga_runs', latest_dir)
        
        print(f"Analyzing latest experiment run: {latest_path}")
        
        df = analyze_experiments(latest_path)
        
        if not df.empty:
            df = df.sort_values(by="threshold").set_index("threshold")
            
            print("\n--- Experiment Results ---")
            print(df)

            # --- Summary ---
            print("\n--- Summary ---")
            reached_zero = df[df["reached_zero_mask"]]

            if not reached_zero.empty:
                print("\nThresholds that successfully reduced mask weight to zero:")
                for threshold in reached_zero.index:
                    print(f"  - {threshold}")
            else:
                print("\nNo experiments successfully reduced the mask weight to zero.")
            
            valid_distances = df["best_manhattan_distance"].dropna()
            if not valid_distances.empty:
                best_run = df.loc[valid_distances.idxmin()]
                print(f"Lowest Manhattan Distance: {best_run['best_manhattan_distance']:.4f} (achieved with threshold {best_run.name})")
            else:
                print("No valid Manhattan distances were recorded for any run.")

        else:
            print("No log files found to analyze.")
