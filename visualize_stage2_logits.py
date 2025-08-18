import torch
from torch.utils.data import DataLoader
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from vae_transformer import TransformerVAE, GridDataset, VOCAB_SIZE

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_categorical_vae.pth'
LATENT_DATA_FILE = 'optimal_latents.json'
DATA_DIR = 'data'
NUM_SAMPLES_TO_PLOT = 4 # Keep this number small to avoid clutter

def annotate_heatmap(im, data):
    """Adds numerical annotations to each cell of a heatmap."""
    # Get the colormap and normalization from the image object
    norm = im.norm
    cmap = im.cmap
    
    # Determine the threshold for text color based on the midpoint of the colormap's range
    threshold = (norm.vmax + norm.vmin) / 2.0
    
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            value = data[r, c]
            # Use the normalization to determine the background color's brightness
            normalized_value = norm(value)
            # Choose text color based on the background brightness
            color = "white" if normalized_value < 0.5 else "black" # Heuristic for 'hot' colormap
            im.axes.text(c, r, f"{value:.1f}", ha="center", va="center", color=color, fontsize=6)

# --- Load Model, Original Data, and Optimal Latents ---
print("Loading resources for logit visualization...")
model = TransformerVAE().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

train_dataset = GridDataset(os.path.join(DATA_DIR, 'train'))

with open(LATENT_DATA_FILE, 'r') as f:
    optimal_latents_data = json.load(f)

# --- Generate Visualization ---
print(f"Generating annotated logit plots for the first {NUM_SAMPLES_TO_PLOT} samples...")
fig, axes = plt.subplots(NUM_SAMPLES_TO_PLOT, 4, figsize=(16, NUM_SAMPLES_TO_PLOT * 4))
fig.suptitle('Stage 2: Annotated Logit Analysis of GA-Optimized Reconstructions', fontsize=18)

with torch.no_grad():
    # First, find the global min and max logits for a consistent color scale
    global_min = float('inf')
    global_max = float('-inf')
    all_logits_data = []

    for i in range(NUM_SAMPLES_TO_PLOT):
        optimal_z = torch.tensor(optimal_latents_data[i]['optimal_z'], dtype=torch.float32).to(DEVICE)
        logits = model.get_logits_from_z(optimal_z.unsqueeze(0)).squeeze(0)
        all_logits_data.append(logits)
        
        current_min = logits.min().item()
        current_max = logits.max().item()
        if current_min < global_min:
            global_min = current_min
        if current_max > global_max:
            global_max = current_max

    for i in range(NUM_SAMPLES_TO_PLOT):
        # Get original ground truth
        _, output_quantized = train_dataset[i]
        
        # Use the pre-calculated logits
        logits = all_logits_data[i]
        
        # Get the final reconstructed image by taking the argmax
        reconstructed_tokens = logits.argmax(dim=-1)
        reconstructed_img = reconstructed_tokens.cpu().numpy().reshape(10, 10).astype(float) / (VOCAB_SIZE - 1)

        # Prepare ground truth for plotting
        truth_img = output_quantized.cpu().numpy().reshape(10, 10).astype(float) / (VOCAB_SIZE - 1)
        
        # Separate the logits for class 0 and class 1
        logits_class_0 = logits[:, 0].cpu().numpy().reshape(10, 10)
        logits_class_1 = logits[:, 1].cpu().numpy().reshape(10, 10)

        # --- Plotting ---
        # Plot Ground Truth
        axes[i, 0].imshow(truth_img, cmap='viridis', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Ground Truth-{i}')
        axes[i, 0].axis('off')

        # Plot GA Reconstruction (Argmax)
        axes[i, 1].imshow(reconstructed_img, cmap='viridis', vmin=0, vmax=1)
        axes[i, 1].set_title(f'GA Recon-{i}')
        axes[i, 1].axis('off')

        # Plot Logits for Class 1 ("On")
        im2 = axes[i, 2].imshow(logits_class_1, cmap='hot', vmin=global_min, vmax=global_max)
        axes[i, 2].set_title(f'Logits for Class 1 ("On")')
        axes[i, 2].axis('off')
        annotate_heatmap(im2, logits_class_1)
        fig.colorbar(im2, ax=axes[i, 2])

        # Plot Logits for Class 0 ("Off")
        im3 = axes[i, 3].imshow(logits_class_0, cmap='hot', vmin=global_min, vmax=global_max)
        axes[i, 3].set_title(f'Logits for Class 0 ("Off")')
        axes[i, 3].axis('off')
        annotate_heatmap(im3, logits_class_0)
        fig.colorbar(im3, ax=axes[i, 3])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('stage2_logit_analysis_annotated.png')
print("Plot saved to 'stage2_logit_analysis_annotated.png'")