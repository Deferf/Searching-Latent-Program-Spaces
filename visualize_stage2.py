import torch
from torch.utils.data import DataLoader
import os
import json
import matplotlib.pyplot as plt
from vae_transformer import TransformerVAE, GridDataset, VOCAB_SIZE

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_categorical_vae.pth'
LATENT_DATA_FILE = 'optimal_latents.json'
DATA_DIR = 'data'
NUM_SAMPLES_TO_PLOT = 5

# --- Load Model, Original Data, and Optimal Latents ---
print("Loading resources for GA results visualization...")
model = TransformerVAE().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Load the original training data to get ground truth outputs
train_dataset = GridDataset(os.path.join(DATA_DIR, 'train'))

# Load the GA-found latents
with open(LATENT_DATA_FILE, 'r') as f:
    optimal_latents_data = json.load(f)

# --- Generate Visualization ---
print(f"Generating plot for the first {NUM_SAMPLES_TO_PLOT} samples...")
fig, axes = plt.subplots(NUM_SAMPLES_TO_PLOT, 3, figsize=(9, NUM_SAMPLES_TO_PLOT * 3))
fig.suptitle('Stage 2: GA-Optimized Latent Vector Reconstructions', fontsize=16)

with torch.no_grad():
    for i in range(NUM_SAMPLES_TO_PLOT):
        # Get original input and ground truth
        input_grid, output_quantized = train_dataset[i]
        
        # Get the corresponding optimal latent vector from the file
        optimal_z = torch.tensor(optimal_latents_data[i]['optimal_z'], dtype=torch.float32).to(DEVICE)
        
        # Decode the optimal z to get the reconstructed image
        reconstructed_img = model.decode_from_z(optimal_z.unsqueeze(0)).squeeze(0)

        # Prepare for plotting
        input_img = input_grid.cpu().numpy().reshape(10, 10)
        truth_img = output_quantized.cpu().numpy().reshape(10, 10).astype(float) / (VOCAB_SIZE - 1)
        reconstructed_img = reconstructed_img.cpu().numpy().reshape(10, 10)

        # Plot Input
        axes[i, 0].imshow(input_img, cmap='Greys', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Input-{i}')
        axes[i, 0].axis('off')

        # Plot Ground Truth
        axes[i, 1].imshow(truth_img, cmap='viridis', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Ground Truth-{i}')
        axes[i, 1].axis('off')

        # Plot GA Reconstruction
        axes[i, 2].imshow(reconstructed_img, cmap='viridis', vmin=0, vmax=1)
        axes[i, 2].set_title(f'GA Recon-{i}')
        axes[i, 2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('stage2_ga_predictions.png')
print("Plot saved to 'stage2_ga_predictions.png'")
