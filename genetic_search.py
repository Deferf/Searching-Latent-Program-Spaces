import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from torch.utils.data import DataLoader
from datetime import datetime

# --- Import necessary components from the VAE script ---
from vae_transformer import TransformerVAE, GridDataset, VOCAB_SIZE

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_categorical_vae.pth'
DATA_DIR = 'data'

# --- 1. Genetic Algorithm (Minimization Version) ---

class GeneticAlgorithm:
    def __init__(self, model, target_quantized, population_size=50, mutation_rate=0.05, crossover_rate=0.8, elitism_size=2):
        self.model = model.to(DEVICE)
        self.model.eval()
        self.target_quantized = target_quantized.to(DEVICE)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_size = elitism_size
        self.latent_dim = model.latent_dim

        self.population = self._initialize_population()
        self.loss_scores = self.calculate_loss()

    def _initialize_population(self):
        """Creates a diverse starting population by encoding the entire training set."""
        all_encodings = []
        dataset = GridDataset(os.path.join(DATA_DIR, 'train'))
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(DEVICE)
                mu, _ = self.model.encode(inputs)
                all_encodings.append(mu)
        
        all_encodings = torch.cat(all_encodings, dim=0)
        indices = torch.randperm(all_encodings.size(0))[:self.population_size]
        return all_encodings[indices]

    def calculate_loss(self):
        """Calculates the Cross-Entropy loss for each individual."""
        with torch.no_grad():
            logits = self.model.get_logits_from_z(self.population)
            target_expanded = self.target_quantized.flatten(0).unsqueeze(0).expand(self.population_size, -1)
            loss = nn.functional.cross_entropy(logits.permute(0, 2, 1), target_expanded, reduction='none').mean(dim=1)
        return loss

    def _selection(self):
        """Selects a parent using tournament selection (minimization)."""
        tournament_size = 5
        participant_indices = random.sample(range(self.population_size), tournament_size)
        # Select the individual with the MINIMUM loss
        best_participant_idx = min(participant_indices, key=lambda i: self.loss_scores[i])
        return self.population[best_participant_idx]

    def _crossover(self, parent1, parent2):
        """Creates a child using arithmetic crossover."""
        if random.random() < self.crossover_rate:
            alpha = random.random()
            child = alpha * parent1 + (1 - alpha) * parent2
            return child
        return parent1.clone()

    def _mutation(self, individual):
        """Applies mutation to each gene of the individual independently."""
        for i in range(self.latent_dim):
            if random.random() < self.mutation_rate:
                noise = torch.randn(1).item() * 0.2
                individual[i] += noise
        return individual

    def run_generation(self):
        """Runs one full generation, including elitism for minimization."""
        # Sort population by loss (ascending)
        sorted_indices = torch.argsort(self.loss_scores, descending=False)
        
        new_population = []
        
        # 1. Elitism: Carry over the best (lowest loss) individuals
        for i in range(self.elitism_size):
            new_population.append(self.population[sorted_indices[i]])
            
        # 2. Create the rest of the new population
        for _ in range(self.population_size - self.elitism_size):
            parent1 = self._selection()
            parent2 = self._selection()
            child = self._crossover(parent1, parent2)
            child = self._mutation(child)
            new_population.append(child)
        
        self.population = torch.stack(new_population)
        self.loss_scores = self.calculate_loss()

    def get_best_individual(self):
        """Returns the best individual (lowest loss) and its loss score."""
        best_index = torch.argmin(self.loss_scores)
        return self.population[best_index], self.loss_scores[best_index]

# --- 2. Visualization Function ---

def plot_ga_progress(model, best_individuals_history, target_image, output_dir):
    """Decodes and plots the progress of the GA's best individuals."""
    filename = os.path.join(output_dir, "ga_progress.png")
    print(f"\nVisualizing GA progress and saving to {filename}...")
    with torch.no_grad():
        decoded_images = model.decode_from_z(torch.stack(best_individuals_history))

    num_images = len(decoded_images) + 1
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle("Evolution of the Best Individual (Minimizing Loss)", fontsize=16)
    axes = axes.flatten()

    axes[0].imshow(target_image.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title("Target")
    axes[0].axis('off')

    for i, img in enumerate(decoded_images):
        ax = axes[i + 1]
        ax.imshow(img.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        gen_num = 0 if i == 0 else (i) * 500
        ax.set_title(f"Gen {gen_num}")
        ax.axis('off')

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    print("Progress visualization saved.")

# --- 3. Main Execution ---

if __name__ == '__main__':
    # --- Create Timestamped Output Directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("ga_runs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    # --- Load Model and Data ---
    model = TransformerVAE()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()

    test_dataset = GridDataset(os.path.join(DATA_DIR, 'test'))
    _, target_output_quantized = test_dataset[5]
    target_image_float = target_output_quantized.view(10, 10).float() / (VOCAB_SIZE - 1)

    # --- Initialize and Run GA ---
    NUM_GENERATIONS = 10000
    POPULATION_SIZE = 50
    
    print(f"Starting GA with direct loss minimization for {NUM_GENERATIONS} generations...")
    ga = GeneticAlgorithm(model, target_output_quantized, population_size=POPULATION_SIZE)

    initial_best_z, initial_best_loss = ga.get_best_individual()
    
    print("Running generations...")
    best_loss_history = []
    best_individuals_history = []

    for gen in range(NUM_GENERATIONS):
        ga.run_generation()
        best_z, best_loss = ga.get_best_individual()
        best_loss_history.append(best_loss.item())
        
        if gen == 0 or (gen + 1) % 500 == 0:
            best_individuals_history.append(best_z)
            print(f"Generation {gen + 1}/{NUM_GENERATIONS}, Best Loss: {best_loss.item():.4f}")
    
    print("Genetic Algorithm finished.")

    final_best_z, final_best_loss = ga.get_best_individual()
    with torch.no_grad():
        initial_best_image = model.decode_from_z(initial_best_z.unsqueeze(0)).squeeze(0)
        final_best_image = model.decode_from_z(final_best_z.unsqueeze(0)).squeeze(0)

    # --- Final Results Visualization ---
    results_path = os.path.join(output_dir, "final_results.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("GA Latent Space Search (Minimizing Loss)", fontsize=16)
    axes[0].imshow(target_image_float.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title("Target Image"); axes[0].axis('off')
    axes[1].imshow(initial_best_image.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title(f"Initial Best (Loss: {initial_best_loss:.2f})"); axes[1].axis('off')
    axes[2].imshow(final_best_image.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title(f"Final Best (Loss: {final_best_loss:.2f})"); axes[2].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(results_path)
    print(f"\nFinal results visualization saved to {results_path}")

    # --- Loss History Plot ---
    loss_history_path = os.path.join(output_dir, "loss_history.png")
    plt.figure(figsize=(10, 5))
    plt.plot(best_loss_history)
    plt.title("Best Loss Over Generations")
    plt.xlabel("Generation"); plt.ylabel("Cross-Entropy Loss"); plt.grid(True)
    plt.savefig(loss_history_path)
    print(f"Loss history plot saved to {loss_history_path}")

    # --- Progress Visualization ---
    plot_ga_progress(model, best_individuals_history, target_image_float, output_dir)