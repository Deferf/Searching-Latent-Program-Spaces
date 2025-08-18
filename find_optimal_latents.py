# find_optimal_latents.py
import torch
from vae_transformer import TransformerVAE, GridDataset
from genetic_search import GeneticAlgorithm
import json
import os
from torch.utils.data import DataLoader

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_categorical_vae.pth'
DATA_DIR = 'data'
OUTPUT_FILE = 'optimal_latents.json'
GA_GENERATIONS = 200 # Number of generations to run for each sample
KLD_WEIGHT = 0.1 # Should match the weight used in the GA script for consistency

# --- Load Model and Data ---
print("Loading VAE model and training data...")
model = TransformerVAE().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

train_dataset = GridDataset(os.path.join(DATA_DIR, 'train'))

# --- Main Loop ---
optimal_latents_data = []
print(f"Finding optimal latents for {len(train_dataset)} training samples...")

for i, (input_grid, target_grid) in enumerate(train_dataset):
    print(f"Processing sample {i+1}/{len(train_dataset)}...")
    
    # For each sample, run a GA to find the best z
    ga = GeneticAlgorithm(
        model, 
        target_grid, 
        population_size=50, 
        mutation_rate=0.1, 
        kld_weight=KLD_WEIGHT
    )
    
    # Run for a set number of generations
    for gen in range(GA_GENERATIONS):
        ga.run_generation()
        
    best_z, best_loss = ga.get_best_individual()
    
    # Store the input and the discovered optimal z
    optimal_latents_data.append({
        'input': input_grid.tolist(),
        'optimal_z': best_z.tolist()
    })

# --- Save the new dataset ---
with open(OUTPUT_FILE, 'w') as f:
    json.dump(optimal_latents_data, f)

print(f"\nSaved optimal latents dataset to {OUTPUT_FILE}")
