import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import math
import matplotlib.pyplot as plt

# --- Configuration ---
VOCAB_SIZE = 2 # Number of discrete pixel intensity levels (0 or 1)

# --- 1. Model Components ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class TransformerVAE(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=256, latent_dim=256, grid_size=10, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.grid_size = grid_size
        self.seq_len = grid_size * grid_size
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.vocab_size = vocab_size

        # --- Embeddings ---
        self.encoder_embedding = nn.Linear(1, d_model) # Encoder still takes continuous-like input
        self.decoder_embedding = nn.Embedding(vocab_size, d_model) # Decoder takes discrete token IDs
        self.pos_encoder = PositionalEncoding(d_model, self.seq_len)
        self.output_projection = nn.Linear(d_model, vocab_size) # Outputs logits over vocab

        # --- Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True, dropout=0.0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # --- VAE Bottleneck ---
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)

        # --- Decoder ---
        self.latent_to_memory = nn.Linear(latent_dim, self.seq_len * d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True, dropout=0.0)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, src):
        src_emb = self.encoder_embedding(src.unsqueeze(-1)) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb.permute(1, 0, 2)).permute(1, 0, 2)
        memory = self.transformer_encoder(src_emb)
        pooled_output = memory[:, 0, :]
        mu = self.fc_mu(pooled_output)
        log_var = self.fc_log_var(pooled_output)
        return mu, log_var

    def decode(self, z, tgt):
        batch_size = z.size(0)
        memory = self.latent_to_memory(z).view(batch_size, self.seq_len, self.d_model)
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb.permute(1, 0, 2)).permute(1, 0, 2)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(z.device)
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = self.output_projection(output)
        return output

    def forward(self, src, tgt):
        mu, log_var = self.encode(src)
        z = self.reparameterize(mu, log_var)
        logits = self.decode(z, tgt)
        return logits, mu, log_var

    @torch.no_grad()
    def _decode_autoregressive(self, z, max_len=100):
        """
        Helper function for autoregressive decoding with KV cache.
        Returns generated token sequence and raw logits.
        """
        self.eval()
        batch_size = z.shape[0]
        device = z.device
        
        memory = self.latent_to_memory(z).view(batch_size, self.seq_len, self.d_model)
        self_attn_kv_cache = [None] * len(self.transformer_decoder.layers)
        
        generated_seq = torch.zeros(batch_size, max_len, dtype=torch.long).to(device)
        all_logits = []
        current_token = torch.zeros(batch_size, 1, dtype=torch.long).to(device)

        for i in range(max_len):
            tgt_emb = self.decoder_embedding(current_token) * math.sqrt(self.d_model)
            pos_encoded_input = (self.pos_encoder.pe[i:i+1] + tgt_emb.permute(1, 0, 2)).permute(1, 0, 2)
            output = pos_encoded_input
            
            for j, layer in enumerate(self.transformer_decoder.layers):
                q, k, v = nn.functional.linear(output, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias).chunk(3, dim=-1)
                
                if self_attn_kv_cache[j] is None:
                    cached_k, cached_v = k, v
                else:
                    prev_k, prev_v = self_attn_kv_cache[j]
                    cached_k = torch.cat([prev_k, k], dim=1)
                    cached_v = torch.cat([prev_v, v], dim=1)
                self_attn_kv_cache[j] = (cached_k, cached_v)

                attn_output, _ = nn.functional.multi_head_attention_forward(
                    query=q.transpose(0,1), key=cached_k.transpose(0,1), value=cached_v.transpose(0,1),
                    embed_dim_to_check=self.d_model, num_heads=layer.self_attn.num_heads,
                    in_proj_weight=torch.empty(0), in_proj_bias=layer.self_attn.in_proj_bias,
                    bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.0,
                    out_proj_weight=layer.self_attn.out_proj.weight, out_proj_bias=layer.self_attn.out_proj.bias,
                    training=self.training, use_separate_proj_weight=True,
                    q_proj_weight=layer.self_attn.in_proj_weight.chunk(3)[0],
                    k_proj_weight=layer.self_attn.in_proj_weight.chunk(3)[1],
                    v_proj_weight=layer.self_attn.in_proj_weight.chunk(3)[2])
                
                output = output + layer.dropout1(attn_output.transpose(0, 1))
                output = layer.norm1(output)
                cross_attn_output, _ = layer.multihead_attn(output, memory, memory, need_weights=False)
                output = output + layer.dropout2(cross_attn_output)
                output = layer.norm2(output)
                ff_output = layer.linear2(layer.dropout(nn.functional.relu(layer.linear1(output))))
                output = output + layer.dropout3(ff_output)
                output = layer.norm3(output)

            logits = self.output_projection(output)
            all_logits.append(logits)
            
            next_token = logits.argmax(dim=-1)
            generated_seq[:, i] = next_token.squeeze()
            current_token = next_token
            
        return generated_seq, torch.cat(all_logits, dim=1)

    @torch.no_grad()
    def decode_from_z(self, z, max_len=100):
        generated_seq, _ = self._decode_autoregressive(z, max_len)
        return generated_seq.view(z.shape[0], self.grid_size, self.grid_size).float() / (self.vocab_size - 1)

    @torch.no_grad()
    def get_logits_from_z(self, z, max_len=100):
        """
        Autoregressively decodes a latent vector z and returns the raw logits.
        This is useful for fitness functions that need to compute loss (e.g., CrossEntropy).
        """
        _, all_logits = self._decode_autoregressive(z, max_len)
        return all_logits

    

# --- 2. Data Loading ---

class GridDataset(Dataset):
    def __init__(self, data_dir, vocab_size=VOCAB_SIZE):
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        self.vocab_size = vocab_size
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'r') as f:
            data = json.load(f)
        
        input_grid = torch.tensor(data['input'], dtype=torch.float32).flatten()
        output_grid_continuous = torch.tensor(data['output'], dtype=torch.float32).flatten()
        
        # Quantize the output grid to integer classes
        output_grid_quantized = torch.round(output_grid_continuous * (self.vocab_size - 1)).long()
        
        return input_grid, output_grid_quantized

# --- 3. Training and Evaluation ---

def vae_loss_function(logits, x, mu, log_var, input_data, mask_weight=1.0, beta=1.0):
    # Calculate per-pixel cross entropy
    CE_per_pixel = nn.functional.cross_entropy(logits.permute(0, 2, 1), x, reduction='none')

    # Create the primary mask for pixels where input and output are both 1
    primary_mask = (input_data.long() & x).bool()

    # Create a weight tensor
    # Start with a base weight of (1 - mask_weight) for all pixels
    weights = torch.full_like(x, 1.0 - mask_weight, dtype=torch.float32)
    
    # Set the weight for the primary mask pixels to 1.0
    weights[primary_mask] = mask_weight

    # Apply the weights to the per-pixel loss
    CE = (CE_per_pixel * weights).sum()

    #KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return CE #+ beta * KLD

def train(model, dataloader, optimizer, device, mask_weight, beta):
    model.train()
    total_loss = 0
    for input_data, output_data in dataloader:
        input_data, output_data = input_data.to(device), output_data.to(device)
        optimizer.zero_grad()
        
        # Prepend a start token (0) for teacher forcing
        decoder_input = torch.cat([torch.zeros(output_data.shape[0], 1, dtype=torch.long).to(device), output_data[:, :-1]], dim=1)
        
        logits, mu, log_var = model(input_data, decoder_input)
        loss = vae_loss_function(logits, output_data, mu, log_var, input_data, mask_weight, beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_data, output_data in dataloader:
            input_data, output_data = input_data.to(device), output_data.to(device)
            decoder_input = torch.cat([torch.zeros(output_data.shape[0], 1, dtype=torch.long).to(device), output_data[:, :-1]], dim=1)
            logits, mu, log_var = model(input_data, decoder_input)
            # Note: mask_weight is not used here for a consistent evaluation metric
            loss = vae_loss_function(logits, output_data, mu, log_var, input_data, mask_weight=0.0, beta=1.0) # Evaluate on full reconstruction
            total_loss += loss.item()
    return total_loss / len(dataloader.dataset)

def calculate_unmasked_loss(model, dataloader, device):
    """Calculates the average loss on pixels NOT in the primary mask."""
    model.eval()
    total_unmasked_loss = 0
    total_pixels = 0
    with torch.no_grad():
        for input_data, output_data in dataloader:
            input_data, output_data = input_data.to(device), output_data.to(device)
            
            decoder_input = torch.cat([torch.zeros(output_data.shape[0], 1, dtype=torch.long).to(device), output_data[:, :-1]], dim=1)
            logits, _, _ = model(input_data, decoder_input)

            CE_per_pixel = nn.functional.cross_entropy(logits.permute(0, 2, 1), output_data, reduction='none')
            
            primary_mask = (input_data.long() & output_data).bool()
            unmasked_pixels = ~primary_mask

            loss = CE_per_pixel[unmasked_pixels].sum()
            
            total_unmasked_loss += loss.item()
            total_pixels += unmasked_pixels.sum().item()

    return total_unmasked_loss / total_pixels if total_pixels > 0 else 0


def calculate_manhattan_distance(model, dataloader, device):
    model.eval()
    total_distance = 0
    with torch.no_grad():
        for input_data, output_data in dataloader:
            input_data = input_data.to(device)
            output_data = output_data.to(device)
            
            # Generate images directly from the latent space of the inputs
            mu, _ = model.encode(input_data)
            
            generated_seq, _ = model._decode_autoregressive(mu)
            generated_outputs_flat = generated_seq.view(-1, 100)

            ground_truth_flat = output_data.view_as(generated_outputs_flat)

            # Calculate Manhattan distance
            total_distance += torch.abs(generated_outputs_flat.long() - ground_truth_flat.long()).sum().item()
            
    return total_distance

def visualize_results(model, dataloader, device, filename='inference_results.png'):
    model.eval()
    inputs, outputs_quantized = next(iter(dataloader))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        # Generate images directly from the latent space of the inputs
        mu, _ = model.encode(inputs)
        generated_outputs = model.decode_from_z(mu)

    inputs = inputs.cpu().numpy().reshape(-1, 10, 10)
    # Convert ground truth back to float for visualization
    outputs = outputs_quantized.cpu().numpy().reshape(-1, 10, 10).astype(float) / (VOCAB_SIZE - 1)
    generated_outputs = generated_outputs.cpu().numpy()

    n_samples = min(inputs.shape[0], 5)
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, n_samples * 3))
    fig.suptitle('Model Inference Results (Categorical)', fontsize=16)
    for i in range(n_samples):
        axes[i, 0].imshow(inputs[i], cmap='Greys', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Input-{i}'); axes[i, 0].axis('off')
        axes[i, 1].imshow(outputs[i], cmap='viridis', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Ground Truth-{i}'); axes[i, 1].axis('off')
        axes[i, 2].imshow(generated_outputs[i], cmap='viridis', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Generated-{i}'); axes[i, 2].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    print(f"Inference visualization saved to {filename}")

if __name__ == '__main__':
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = 'data'
    MAX_EPOCHS = 2000
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 8
    MODEL_SAVE_PATH = 'best_categorical_vae.pth'
    DO_TRAIN = True
    
    print(f"Using device: {DEVICE}")

    # --- Data Loaders ---
    train_dataset = GridDataset(os.path.join(DATA_DIR, 'train'))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = GridDataset(os.path.join(DATA_DIR, 'validation'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model Initialization ---
    model = TransformerVAE().to(DEVICE)

    if DO_TRAIN:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

        best_val_manhattan = float('inf')
        mask_weight = 1.0
        epochs_since_improvement = 0
        patience = 10
        mask_reduction_enabled = False
        
        # KL Annealing parameters
        beta = 0.0
        kl_anneal_epochs = 50
        
        # Adaptive mask weight reduction parameters
        SCALING_FACTOR = 0.005
        EPSILON = 1e-6
        MIN_DECREMENT = 0.001
        MAX_DECREMENT = 0.05


        print("Starting training with adaptive weighted mask, KL annealing, and LR scheduler...")
        for epoch in range(1, MAX_EPOCHS + 1):
            # Update beta for KL annealing
            beta = min(1.0, epoch / kl_anneal_epochs)

            train_loss = train(model, train_loader, optimizer, DEVICE, mask_weight, beta)
            
            if not mask_reduction_enabled and train_loss < 1.0:
                print(f'  -> Train loss below 1.0. Mask weight reduction is now enabled.')
                mask_reduction_enabled = True

            # --- Validation Step ---
            train_manhattan_distance = calculate_manhattan_distance(model, train_loader, DEVICE)
            val_manhattan_distance = calculate_manhattan_distance(model, val_loader, DEVICE)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}/{MAX_EPOCHS}, Train Loss: {train_loss:.4f}, Train Manhattan: {train_manhattan_distance:.2f}, Val Manhattan: {val_manhattan_distance:.2f}, Mask Weight: {mask_weight:.3f}, Beta: {beta:.3f}, LR: {current_lr:.6f}')

            if val_manhattan_distance < best_val_manhattan:
                best_val_manhattan = val_manhattan_distance
                epochs_since_improvement = 0
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f'  -> New best Val Manhattan: {best_val_manhattan:.2f}. Model saved.')
            else:
                epochs_since_improvement += 1

            if val_manhattan_distance == 0:
                print(f'\nPerfect reconstruction on Validation Set achieved at epoch {epoch}. Stopping training.')
                break

            if mask_reduction_enabled and epochs_since_improvement >= patience:
                print(f'  -> Val Manhattan did not improve for {patience} epochs.')
                
                # Calculate derivative-based decrement
                unmasked_loss = calculate_unmasked_loss(model, train_loader, DEVICE)
                # The unmasked_loss is a proxy for the gradient of the loss w.r.t. the mask weight
                decrement_size = SCALING_FACTOR / (unmasked_loss + EPSILON)
                decrement_size = max(MIN_DECREMENT, min(decrement_size, MAX_DECREMENT))
                
                print(f'  -> Unmasked Loss (Proxy for Grad): {unmasked_loss:.4f}, Calculated Decrement: {decrement_size:.4f}')
                
                mask_weight = max(0.0, mask_weight - decrement_size)
                print(f'  -> New mask weight: {mask_weight:.3f}.')

                epochs_since_improvement = 0 # Reset patience
                mask_reduction_enabled = False
                print(f'  -> Mask weight reduction is now disabled until train loss is below 1.0 again.')
            
            # Step the scheduler
            scheduler.step(val_manhattan_distance)

        if epoch == MAX_EPOCHS:
            print(f'\nFinished training after {MAX_EPOCHS} epochs.')
            print(f"Best Validation Set Manhattan Distance: {best_val_manhattan:.4f}")
        
        # Save the final model state
        final_model_path = 'final_vae_transformer.pth'
        torch.save(model.state_dict(), final_model_path)
        print(f"\nFinal model weights saved to {final_model_path}")


    # --- Evaluation ---
    print("\n--- Evaluating Best Model ---")
    best_model = TransformerVAE().to(DEVICE)
    best_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    train_manhattan = calculate_manhattan_distance(best_model, train_loader, DEVICE)
    val_manhattan = calculate_manhattan_distance(best_model, val_loader, DEVICE)
    
    print(f'Best Model - Training Set Manhattan Distance: {train_manhattan}')
    print(f'Best Model - Validation Set Manhattan Distance: {val_manhattan}')

    # --- Visualization of Best Model ---
    print("\n--- Generating Visualization on Validation Set (Best Model) ---")
    visualize_results(best_model, val_loader, DEVICE, filename='best_model_inference_results.png')

    # --- Evaluation of Final Model ---
    print("\n--- Evaluating Final Model ---")
    final_model = TransformerVAE().to(DEVICE)
    final_model.load_state_dict(torch.load(final_model_path))

    final_train_manhattan = calculate_manhattan_distance(final_model, train_loader, DEVICE)
    final_val_manhattan = calculate_manhattan_distance(final_model, val_loader, DEVICE)

    print(f'Final Model - Training Set Manhattan Distance: {final_train_manhattan}')
    print(f'Final Model - Validation Set Manhattan Distance: {final_val_manhattan}')

    # --- Visualization of Final Model ---
    print("\n--- Generating Visualization on Training Set (Final Model) ---")
    visualize_results(final_model, train_loader, DEVICE, filename='final_model_train_inference_results.png')
    print("\n--- Generating Visualization on Validation Set (Final Model) ---")
    visualize_results(final_model, val_loader, DEVICE, filename='final_model_val_inference_results.png')
