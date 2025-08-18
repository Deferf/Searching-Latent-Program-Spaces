import torch
import time
from vae_transformer import TransformerVAE

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
GRID_SIZE = 10
SEQ_LEN = GRID_SIZE * GRID_SIZE
MAX_GEN_LEN = 100

# Check if CUDA is available and selected
if DEVICE.type == 'cuda':
    print("CUDA is available. Running on GPU.")
else:
    print("CUDA not available. Running on CPU.")

print(f"Using device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Sequence length: {SEQ_LEN}")

# --- Model Initialization ---
print("Initializing model...")
model = TransformerVAE().to(DEVICE)
model.eval()

# --- Dummy Data ---
print("Creating dummy input data...")
# Create a random input tensor
src = torch.rand(BATCH_SIZE, SEQ_LEN).to(DEVICE)

# --- Benchmarking ---
print("\nStarting inference benchmark...")

# GPU warm-up
if DEVICE.type == 'cuda':
    print("Warming up GPU...")
    for _ in range(5):
        _ = model.generate(src, max_len=MAX_GEN_LEN)
    torch.cuda.synchronize()

start_time = time.time()

# Run the generation
with torch.no_grad():
    generated_output = model.generate(src, max_len=MAX_GEN_LEN)

# Synchronize for accurate timing on GPU
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()

end_time = time.time()
inference_time = end_time - start_time

print("Inference finished.")
print(f"Generated output shape: {generated_output.shape}")
print(f"Time taken for a single inference: {inference_time:.4f} seconds")