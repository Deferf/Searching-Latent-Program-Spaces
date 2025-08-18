import numpy as np
import matplotlib.pyplot as plt
import os
import json

def generate_all_data(grid_size=10, pattern_size=3):
    """
    Generates all possible input and output grids in a fixed order.
    """
    all_input_grids = []
    all_output_grids = []
    offset = pattern_size // 2

    # Use a fixed seed to generate one constant random pattern
    np.random.seed(42)
    constant_random_pattern = np.random.rand(pattern_size, pattern_size)

    dot_positions = []
    for r in range(offset, grid_size - offset):
        for c in range(offset, grid_size - offset):
            dot_positions.append((r, c))

    # Sort positions to have a consistent, raster-scan order
    dot_positions.sort()

    for r, c in dot_positions:
        input_grid = np.zeros((grid_size, grid_size))
        input_grid[r, c] = 1
        all_input_grids.append(input_grid)

        output_grid = np.zeros((grid_size, grid_size))
        output_grid[r - offset:r + offset + 1, c - offset:c + offset + 1] = constant_random_pattern
        all_output_grids.append(output_grid)

    return np.array(all_input_grids), np.array(all_output_grids)

def get_datasets_no_shuffle(grid_size=10, pattern_size=3, train_split=0.8, val_split=0.1):
    """
    Generates and splits the data sequentially into training, validation, and test sets.
    """
    inputs, outputs = generate_all_data(grid_size, pattern_size)
    total_samples = len(inputs)

    train_end = int(total_samples * train_split)
    val_end = train_end + int(total_samples * val_split)

    datasets = {
        'train': (inputs[:train_end], outputs[:train_end]),
        'validation': (inputs[train_end:val_end], outputs[train_end:val_end]),
        'test': (inputs[val_end:], outputs[val_end:])
    }
    
    split_indices = {
        'train': list(range(train_end)),
        'validation': list(range(train_end, val_end)),
        'test': list(range(val_end, total_samples))
    }
    
    return datasets, split_indices

def save_data_to_json(datasets, base_dir='data'):
    """
    Saves the datasets to JSON files in a structured directory.
    """
    if os.path.exists(base_dir):
        import shutil
        shutil.rmtree(base_dir)
        
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"\nSaving data to '{base_dir}' directory...")
    for split_name, (inputs, outputs) in datasets.items():
        split_dir = os.path.join(base_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for i in range(len(inputs)):
            file_path = os.path.join(split_dir, f'sample_{i:03d}.json')
            data_to_save = {
                'input': inputs[i].tolist(),
                'output': outputs[i].tolist()
            }
            with open(file_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
    print("Data saved successfully.")


def plot_data_splits_grid(datasets, split_indices, filename='data_visualization.png'):
    """
    Plots all input-output pairs in a grid, sectioned by dataset split.
    """
    total_samples = sum(len(inputs) for inputs, _ in datasets.values())
    
    # Determine grid size - aim for a roughly square layout of pairs
    # Each pair takes 2 columns (input, output)
    ncols_pairs = int(np.ceil(np.sqrt(total_samples)))
    ncols = ncols_pairs * 2
    nrows = int(np.ceil(total_samples / ncols_pairs))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.2))
    fig.suptitle('Train, Validation, and Test Data Splits (Grid View)', fontsize=20, y=1.0)
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Hide all axes initially, they will be turned on as needed
    for ax in axes:
        ax.axis('off')

    current_sample_idx = 0
    for split_name in ['train', 'validation', 'test']:
        inputs, outputs = datasets[split_name]
        
        if len(inputs) == 0:
            continue
            
        # Add a title for the split section
        # Place it before the first sample of the split
        ax_title_pos = current_sample_idx * 2
        if ax_title_pos < len(axes):
            fig.text(axes[ax_title_pos].get_position().x0, 
                     axes[ax_title_pos].get_position().y1 + 0.01, 
                     f'{split_name.upper()} SET', 
                     fontsize=16, fontweight='bold', ha='left', va='bottom')

        for i in range(len(inputs)):
            original_index = split_indices[split_name][i]
            ax_idx = (current_sample_idx) * 2
            
            if ax_idx + 1 >= len(axes):
                print("Warning: Not all samples could be plotted in the grid.")
                break

            # Plot Input
            ax_input = axes[ax_idx]
            ax_input.imshow(inputs[i], cmap='Greys', vmin=0, vmax=1)
            ax_input.set_title(f'Input {original_index}')
            ax_input.axis('on')
            ax_input.set_xticks([])
            ax_input.set_yticks([])

            # Plot Output
            ax_output = axes[ax_idx + 1]
            ax_output.imshow(outputs[i], cmap='viridis', vmin=0, vmax=1)
            ax_output.set_title(f'Output {original_index}')
            ax_output.axis('on')
            ax_output.set_xticks([])
            ax_output.set_yticks([])
            
            current_sample_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename)
    print(f"\nGrid visualization saved to {filename}")


if __name__ == '__main__':
    # 1. Generate and split the data without shuffling
    datasets, split_indices = get_datasets_no_shuffle()

    print("Dataset sizes:")
    for name, (inputs, outputs) in datasets.items():
        print(f"  {name.capitalize()}:")
        print(f"    Inputs shape: {inputs.shape}")
        print(f"    Outputs shape: {outputs.shape}")

    # 2. Save the data to JSON files
    save_data_to_json(datasets)

    # 3. Plot all the data and save to a file
    plot_data_splits_grid(datasets, split_indices)
