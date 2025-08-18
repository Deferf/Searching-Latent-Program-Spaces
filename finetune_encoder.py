# finetune_encoder.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from vae_transformer import TransformerVAE

# --- Custom Dataset for the optimal latents data ---
class OptimalZDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        # Ensure tensors are flattened correctly
        return torch.tensor(item['input'], dtype=torch.float32).flatten(), torch.tensor(item['optimal_z'], dtype=torch.float32)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_categorical_vae.pth'
FINETUNED_MODEL_PATH = 'finetuned_vae.pth'
LATENT_DATA_FILE = 'optimal_latents.json'
LEARNING_RATE = 1e-5 # Strategy 4: Use a low learning rate for fine-tuning
BATCH_SIZE = 16
EPOCHS = 50

# --- Load Model ---
print(f"Loading base model from {MODEL_PATH}...")
model = TransformerVAE().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# --- FREEZE DECODER WEIGHTS ---
print("Freezing decoder weights for encoder-only training...")
for param in model.decoder_embedding.parameters():
    param.requires_grad = False
for param in model.transformer_decoder.parameters():
    param.requires_grad = False
for param in model.latent_to_memory.parameters():
    param.requires_grad = False
for param in model.output_projection.parameters():
    param.requires_grad = False

# --- Prepare for Training ---
print(f"Loading optimal latents dataset from {LATENT_DATA_FILE}...")
try:
    finetune_dataset = OptimalZDataset(LATENT_DATA_FILE)
    finetune_loader = DataLoader(finetune_dataset, batch_size=BATCH_SIZE, shuffle=True)
except FileNotFoundError:
    print(f"Error: Latent data file not found at {LATENT_DATA_FILE}")
    print("Please run 'find_optimal_latents.py' first to generate the dataset.")
    exit()


# We only pass the encoder's parameters to the optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# --- Fine-Tuning Loop ---
print("Starting encoder fine-tuning...")
model.train() # Set model to training mode
for epoch in range(EPOCHS):
    total_loss = 0
    for input_grids, target_zs in finetune_loader:
        input_grids = input_grids.to(DEVICE)
        target_zs = target_zs.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Get the encoder's output
        mu, _ = model.encode(input_grids)
        
        # Calculate loss against the GA's optimal z
        loss = loss_fn(mu, target_zs)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(finetune_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, MSE Loss: {avg_loss:.6f}")

# --- Save the fine-tuned model ---
torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
print(f"\nSaved fine-tuned model to {FINETUNED_MODEL_PATH}")
