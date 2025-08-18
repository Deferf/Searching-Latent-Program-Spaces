# stage2_optimized_v2.py
import torch
import torch.nn as nn
import json
import os
from torch.utils.data import DataLoader
import time
import cProfile
import pstats
import io

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
        self.fitness_scores = None # Initialize fitness scores

    def calculate_fitness(self, population):
        """Calculates the fitness for a given population."""
        with torch.no_grad():
            pop_reshaped = population.view(-1, self.latent_dim)
            logits = self.model.get_logits_from_z(pop_reshaped)
            logits = logits.view(self.num_samples, -1, self.model.seq_len, VOCAB_SIZE)
            
            targets_expanded = self.target_grids.unsqueeze(1).expand(-1, population.shape[1], -1)
            
            ce_logits = logits.permute(0, 1, 3, 2).reshape(-1, self.model.vocab_size, self.model.seq_len)
            ce_targets = targets_expanded.reshape(-1, self.model.seq_len)

            ce_loss = nn.functional.cross_entropy(
                ce_logits,
                ce_targets,
                reduction='none'
            ).mean(dim=1).view(self.num_samples, -1)

            kld = 0.5 * torch.sum(population.pow(2), dim=2)
            total_loss = ce_loss + self.kld_weight * kld
        return total_loss

    def run_generation(self):
        """Runs one full generation of the GA for all samples in parallel."""
        if self.fitness_scores is None:
            self.fitness_scores = self.calculate_fitness(self.population)

        # 1. Elitism
        _, elite_indices = torch.topk(self.fitness_scores, self.elitism_size, largest=False, sorted=True)
        elites = torch.gather(self.population, 1, elite_indices.unsqueeze(-1).expand(-1, -1, self.latent_dim))
        elite_fitness = torch.gather(self.fitness_scores, 1, elite_indices)

        # 2. Selection (Tournament)
        num_to_breed = self.population_size - self.elitism_size
        parent1 = self._tournament_selection(self.fitness_scores, num_to_breed)
        parent2 = self._tournament_selection(self.fitness_scores, num_to_breed)

        # 3. Crossover
        children = self._crossover(parent1, parent2)

        # 4. Mutation
        mutated_children = self._mutation(children)
        
        # 5. Calculate fitness ONLY for the new children
        children_fitness = self.calculate_fitness(mutated_children)

        # 6. Form new population and fitness scores
        self.population = torch.cat([elites, mutated_children], dim=1)
        self.fitness_scores = torch.cat([elite_fitness, children_fitness], dim=1)


    def _tournament_selection(self, fitness_scores, num_selections, tournament_size=5):
        participant_indices = torch.randint(0, self.population_size, 
                                            (self.num_samples, num_selections, tournament_size), 
                                            device=self.device)
        participant_fitness = torch.gather(fitness_scores.unsqueeze(1).expand(-1, num_selections, -1), 
                                           2, 
                                           participant_indices)
        winner_local_indices = torch.argmin(participant_fitness, dim=2)
        winner_indices = torch.gather(participant_indices, 2, winner_local_indices.unsqueeze(-1)).squeeze(-1)
        winners = torch.gather(self.population, 1, winner_indices.unsqueeze(-1).expand(-1, -1, self.latent_dim))
        return winners

    def _crossover(self, parent1, parent2):
        mask = (torch.rand(self.num_samples, parent1.shape[1], 1, device=self.device) < self.crossover_rate).float()
        alpha = torch.rand(self.num_samples, parent1.shape[1], 1, device=self.device)
        children = alpha * parent1 + (1.0 - alpha) * parent2
        return mask * children + (1.0 - mask) * parent1

    def _mutation(self, children):
        mask = (torch.rand_like(children) < self.mutation_rate).float()
        noise = torch.randn_like(children) * 0.2
        return children + mask * noise

    def get_best_individuals(self):
        """Returns the single best individual for each sample."""
        if self.fitness_scores is None:
            self.fitness_scores = self.calculate_fitness(self.population)
        best_indices = torch.argmin(self.fitness_scores, dim=1)
        best_individuals = torch.gather(self.population, 1, best_indices.view(-1, 1, 1).expand(-1, -1, self.latent_dim)).squeeze(1)
        return best_individuals

# --- Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = 'best_categorical_vae.pth'
    DATA_DIR = 'data'
    OUTPUT_FILE = 'optimal_latents.json'
    GA_GENERATIONS = 200 # Reduced for profiling
    POPULATION_SIZE = 100
    KLD_WEIGHT = 0.1
    
    print(f"Using device: {DEVICE}")

    # --- Load Model and Data ---
    print("Loading VAE model and training data...")
    model = TransformerVAE().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    train_dataset = GridDataset(os.path.join(DATA_DIR, 'train'))
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    all_inputs, all_targets = next(iter(train_loader))
    all_inputs = all_inputs.to(DEVICE)
    all_targets = all_targets.to(DEVICE)
    num_samples = all_inputs.shape[0]

    # --- Initialize Population ---
    print(f"Initializing population for {num_samples} samples...")
    with torch.no_grad():
        initial_mu, _ = model.encode(all_inputs)
    
    initial_population = initial_mu.unsqueeze(1).expand(-1, POPULATION_SIZE, -1)
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

    # --- Profiling Setup ---
    profiler = cProfile.Profile()
    
    start_time = time.time()
    
    # --- Run with Profiler ---
    profiler.enable()
    for gen in range(GA_GENERATIONS):
        ga.run_generation()
        if (gen + 1) % 100 == 0:
            print(f"Generation {gen + 1}/{GA_GENERATIONS}...")
    profiler.disable()

    end_time = time.time()
    
    print(f"\nGA finished in {end_time - start_time:.2f} seconds.")

    # --- Print Profiling Results ---
    print("\n--- Profiling Results ---")
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(30)
    print(s.getvalue())


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
