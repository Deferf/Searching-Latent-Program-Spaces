import torch
from torch.utils.data import DataLoader
import os
from vae_transformer import TransformerVAE, GridDataset, visualize_results

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'finetuned_vae.pth' # Use the fine-tuned model
DATA_DIR = 'data'

# --- Load Model and Data ---
print("Loading fine-tuned VAE model and training data for visualization...")
model = TransformerVAE().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

train_dataset = GridDataset(os.path.join(DATA_DIR, 'train'))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# --- Generate Visualization ---
print("Generating plot of predictions on the training set from the fine-tuned model...")
visualize_results(model, train_loader, DEVICE, filename='stage3_finetuned_predictions.png')
print("Plot saved to 'stage3_finetuned_predictions.png'")
