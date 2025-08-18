# stage2_optimized.py
import torch
import torch.nn as nn
import json
import os
from torch.utils.data import DataLoader
import time

from vae_transformer import TransformerVAE, GridDataset, VOCAB_SIZE

class BatchedGeneticAlgorithm:
    """
    A Genetic Algorithm implementation that is fully vectorized to run on the GPU.
    It processes a batch of GAs in parallel, one for each training sample.
    """
    def __init__(self, model, initial_population, target_grids, population_size=100, mutation_rate=0.05, crossover_rate=0.8, elitism_size=4, kld_weight=0.1, device='cuda'):
        self.model = model
        self.population = initial_population.to(device) # Shape: (num_samples, pop_size, latent_dim)
        self.target_grids = target_grids.to(device)     # Shape: (num_samples, seq_len)
        self.num_samples = initial_population.shape[0]
        self.population_size = population_size
        self.latent_dim = initial_population.shape[2]
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_size = elitism_size
        self.kld_weight = kld_weight
        self.device = device

    def calculate_fitness(self):
        """
        Calculates the fitness for a given population of (mu, log_var) pairs.
        The fitness is the negative VAE loss (reconstruction + KLD).
        """
        with torch.no_grad():
            # The population consists of mu and log_var concatenated
            mu, log_var = self.population.chunk(2, dim=-1)

            # 1. Reparameterization Trick to get z
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # 2. Reconstruction Loss (Cross-Entropy)
            z_reshaped = z.reshape(-1, z.shape[-1])
            logits = self.model.get_logits_from_z(z_reshaped)
            logits = logits.view(self.num_samples, self.population_size, -1, VOCAB_SIZE)
            
            targets_expanded = self.target_grids.unsqueeze(1).expand(-1, self.population_size, -1)
            
            ce_logits = logits.permute(0, 1, 3, 2).reshape(-1, self.model.vocab_size, self.model.seq_len)
            ce_targets = targets_expanded.reshape(-1, self.model.seq_len)

            ce_loss = nn.functional.cross_entropy(
                ce_logits,
                ce_targets,
                reduction='none'
            ).mean(dim=1).view(self.num_samples, -1)

            # 3. KL Divergence (matches the VAE training loss)
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=2)
            
            # 4. Total VAE Loss
            total_loss = ce_loss + self.kld_weight * kld
        return total_loss

    def run_generation(self):
        """Runs one full generation of the GA for all samples in parallel."""
        fitness_scores = self.calculate_fitness() # Shape: (num_samples, pop_size)

        # 1. Elitism
        # Find the top 'elitism_size' individuals for each sample
        _, elite_indices = torch.topk(fitness_scores, self.elitism_size, largest=False, sorted=False)
        elites = torch.gather(self.population, 1, elite_indices.unsqueeze(-1).expand(-1, -1, self.latent_dim))

        # 2. Selection (Tournament)
        num_to_breed = self.population_size - self.elitism_size
        
        # Select parents for all children of all samples at once
        parent1 = self._tournament_selection(fitness_scores, num_to_breed)
        parent2 = self._tournament_selection(fitness_scores, num_to_breed)

        # 3. Crossover
        children = self._crossover(parent1, parent2)

        # 4. Mutation
        mutated_children = self._mutation(children)

        # 5. Form new population
        self.population = torch.cat([elites, mutated_children], dim=1)

    def _tournament_selection(self, fitness_scores, num_selections, tournament_size=5):
        # Generate tournament participants for all samples and all selections
        participant_indices = torch.randint(0, self.population_size, 
                                            (self.num_samples, num_selections, tournament_size), 
                                            device=self.device)
        
        # Get fitness of all participants
        participant_fitness = torch.gather(fitness_scores.unsqueeze(1).expand(-1, num_selections, -1), 
                                           2, 
                                           participant_indices)
        
        # Find the winner of each tournament (index within the tournament)
        winner_local_indices = torch.argmin(participant_fitness, dim=2)
        
        # Get the population index of each winner
        winner_indices = torch.gather(participant_indices, 2, winner_local_indices.unsqueeze(-1)).squeeze(-1)
        
        # Retrieve the winning individuals
        winners = torch.gather(self.population, 1, winner_indices.unsqueeze(-1).expand(-1, -1, self.latent_dim))
        return winners

    def _crossover(self, parent1, parent2):
        # Create crossover mask
        mask = (torch.rand(self.num_samples, parent1.shape[1], 1, device=self.device) < self.crossover_rate).float()
        # Create random blending factor
        alpha = torch.rand(self.num_samples, parent1.shape[1], 1, device=self.device)
        
        # Perform crossover
        children = alpha * parent1 + (1.0 - alpha) * parent2
        
        # Apply mask: where mask is 0, use parent1, where 1, use the blended child
        return mask * children + (1.0 - mask) * parent1

    def _mutation(self, children):
        # Create mutation mask
        mask = (torch.rand_like(children) < self.mutation_rate).float()
        # Create noise
        noise = torch.randn_like(children) * 0.2 # Std dev of noise
        
        # Apply mutation
        return children + mask * noise

    def get_best_individuals(self):
        """Returns the single best individual for each sample."""
        fitness_scores = self.calculate_fitness()
        best_indices = torch.argmin(fitness_scores, dim=1)
        best_individuals = torch.gather(self.population, 1, best_indices.view(-1, 1, 1).expand(-1, -1, self.latent_dim)).squeeze(1)
        return best_individuals

# --- Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = 'best_categorical_vae.pth'
    DATA_DIR = 'data'
    OUTPUT_FILE = 'optimal_latents.json'
    GA_GENERATIONS = 2000 # Increased generations due to performance improvement
    POPULATION_SIZE = 100
    KLD_WEIGHT = 0.1
    
    print(f"Using device: {DEVICE}")

    # --- Load Model and Data ---
    print("Loading VAE model and training data...")
    model = TransformerVAE().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    train_dataset = GridDataset(os.path.join(DATA_DIR, 'train'))
    # Load the entire dataset into a single batch on the GPU
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    all_inputs, all_targets = next(iter(train_loader))
    all_inputs = all_inputs.to(DEVICE)
    all_targets = all_targets.to(DEVICE)
    num_samples = all_inputs.shape[0]

    # --- Initialize Population ---
    print(f"Initializing population for {num_samples} samples...")
    with torch.no_grad():
        initial_mu, initial_log_var = model.encode(all_inputs)
    
    # Each individual in the population is a concatenation of mu and log_var
    initial_params = torch.cat([initial_mu, initial_log_var], dim=-1)
    
    # Create the initial population by taking the encoder's output and adding noise
    initial_population = initial_params.unsqueeze(1).expand(-1, POPULATION_SIZE, -1)
    noise = torch.randn_like(initial_population) * 0.1
    initial_population = initial_population + noise

    # --- Initialize and Run Batched GA ---
    print(f"Starting batched GA for {GA_GENERATIONS} generations...")
    ga = BatchedGeneticAlgorithm(
        model,
        initial_population,
        all_targets,
        population_size=POPULATION_SIZE,
        kld_weight=KLD_WEIGHT,
        device=DEVICE
    )

    start_time = time.time()
    for gen in range(GA_GENERATIONS):
        ga.run_generation()
        if (gen + 1) % 100 == 0:
            print(f"Generation {gen + 1}/{GA_GENERATIONS}...")
    end_time = time.time()
    
    print(f"\nGA finished in {end_time - start_time:.2f} seconds.")

    # --- Save Results ---
    print("Extracting best individuals and saving to file...")
    best_latents = ga.get_best_individuals().cpu().tolist()

    optimal_latents_data = []
    for i in range(num_samples):
        optimal_latents_data.append({
            'input': all_inputs[i].cpu().tolist(),
            'optimal_z': best_latents[i]
        })

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(optimal_latents_data, f)

    print(f"Saved optimal latents dataset to {OUTPUT_FILE}")
