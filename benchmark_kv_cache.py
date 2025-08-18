import torch
import time
from vae_transformer import TransformerVAE

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
GRID_SIZE = 10
SEQ_LEN = GRID_SIZE * GRID_SIZE
MAX_GEN_LEN = 100
WARMUP_RUNS = 5
BENCHMARK_RUNS = 20

# Check if CUDA is available and selected
if DEVICE.type == 'cuda':
    print("CUDA is available. Running on GPU.")
else:
    print("CUDA not available. Running on CPU.")

print(f"Using device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Sequence length: {SEQ_LEN}")
print(f"Generating {MAX_GEN_LEN} tokens.")

# --- Model Initialization ---
print("\nInitializing model...")
model = TransformerVAE().to(DEVICE)
model.eval()

# --- Dummy Data ---
print("Creating dummy input data...")
# Create a random latent vector
z = torch.randn(BATCH_SIZE, model.latent_dim).to(DEVICE)

# --- Benchmarking ---

def benchmark_method(method_name):
    print(f"--- Benchmarking {method_name} ---")
    method_to_run = getattr(model, method_name)
    
    # GPU warm-up
    if DEVICE.type == 'cuda':
        print("Warming up GPU...")
        for _ in range(WARMUP_RUNS):
            _ = method_to_run(z, max_len=MAX_GEN_LEN)
        torch.cuda.synchronize()

    # Run the benchmark
    total_time = 0
    for _ in range(BENCHMARK_RUNS):
        start_time = time.time()
        with torch.no_grad():
            _ = method_to_run(z, max_len=MAX_GEN_LEN)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        total_time += (end_time - start_time)

    avg_time = total_time / BENCHMARK_RUNS
    print(f"Average inference time: {avg_time:.4f} seconds")
    return avg_time

# Run benchmarks
slow_time = benchmark_method('decode_from_z_slow')
fast_time = benchmark_method('decode_from_z')

# --- Results ---
print("\n--- Benchmark Results ---")
print(f"Original method ('decode_from_z_slow'): {slow_time:.4f} seconds")
print(f"Optimized method ('decode_from_z'):    {fast_time:.4f} seconds")

if fast_time < slow_time:
    speedup = slow_time / fast_time
    print(f"\nThe KV cache implementation is {speedup:.2f}x faster.")
else:
    print("\nThe KV cache implementation did not result in a speedup.")
