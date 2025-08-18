import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import math
import matplotlib.pyplot as plt

# --- 1. Model Components ---

class PositionalEncoding(nn.Module):
    """Adds positional information to the input embeddings."""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return x

class TransformerVAE(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256, latent_dim=16, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        self.seq_len = grid_size * grid_size
        self.latent_dim = latent_dim
        self.d_model = d_model

        # --- Input/Output Embeddings ---
        self.input_embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, self.seq_len)
        self.output_projection = nn.Linear(d_model, 1)

        # --- Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # --- VAE Bottleneck ---
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)

        # --- Decoder ---
        self.latent_to_memory = nn.Linear(latent_dim, self.seq_len * d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # --- Autoregressive Generation ---
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, src):
        # src shape: (batch_size, seq_len)
        src = self.input_embedding(src.unsqueeze(-1)) * math.sqrt(self.d_model) # (batch, seq, d_model)
        src = self.pos_encoder(src.permute(1, 0, 2)).permute(1, 0, 2) # Apply PE
        memory = self.transformer_encoder(src)
        
        # Use the representation of the first token as the aggregate
        pooled_output = memory[:, 0, :]
        mu = self.fc_mu(pooled_output)
        log_var = self.fc_log_var(pooled_output)
        return mu, log_var

    def decode(self, z, tgt):
        # z shape: (batch_size, latent_dim)
        # tgt shape: (batch_size, seq_len)
        
        # Project latent vector to be the memory for the decoder
        memory = self.latent_to_memory(z).view(-1, self.seq_len, self.d_model)

        # Prepare target for decoder (teacher forcing)
        tgt_emb = self.input_embedding(tgt.unsqueeze(-1)) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb.permute(1, 0, 2)).permute(1, 0, 2)
        
        # Generate a causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_len).to(z.device)
        
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = self.output_projection(output)
        return output.squeeze(-1)

    def forward(self, src, tgt):
        mu, log_var = self.encode(src)
        z = self.reparameterize(mu, log_var)
        recon_tgt = self.decode(z, tgt)
        return recon_tgt, mu, log_var

    @torch.no_grad()
    def generate(self, src, max_len=100):
        self.eval()
        batch_size = src.shape[0]
        device = src.device

        mu, log_var = self.encode(src)
        z = mu # Use mean for generation
        
        memory = self.latent_to_memory(z).view(batch_size, self.seq_len, self.d_model)
        
        # Start with the start token
        generated_seq = torch.zeros(batch_size, max_len).to(device)
        
        # The `memory` comes from the encoder and has a fixed sequence length.
        # The `tgt` for the decoder grows one token at a time.
        # The decoder's self-attention mask (`tgt_mask`) should match the `tgt` length,
        # while the cross-attention uses the full `memory`. This is the standard
        # behavior and the implementation was correct, but the error message
        # was misleading. The actual issue is that `batch_first=True` in the
        # decoder layer requires the mask to be 3D.
        
        # Start with a single "zero" token as the initial input
        generated_seq_emb = self.input_embedding(torch.zeros(batch_size, 1, 1).to(device)) * math.sqrt(self.d_model)

        for i in range(max_len):
            # Apply positional encoding to the sequence generated so far
            pos_encoded_input = self.pos_encoder(generated_seq_emb.permute(1, 0, 2)).permute(1, 0, 2)
            
            # Create the causal mask for the decoder's self-attention
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(pos_encoded_input.size(1)).to(device)
            
            output = self.transformer_decoder(pos_encoded_input, memory, tgt_mask=tgt_mask)
            
            # Get the prediction for the very last token in the sequence
            last_token_output = self.output_projection(output[:, -1, :])
            
            # Store the raw output value
            generated_seq[:, i] = last_token_output.squeeze(-1)
            
            # Prepare the input for the next step by appending the new prediction
            new_token_emb = self.input_embedding(last_token_output.unsqueeze(-1)) * math.sqrt(self.d_model)
            generated_seq_emb = torch.cat([generated_seq_emb, new_token_emb], dim=1)

        return generated_seq.view(batch_size, self.grid_size, self.grid_size)


# --- 2. Data Loading ---

class GridDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'r') as f:
            data = json.load(f)
        input_grid = torch.tensor(data['input'], dtype=torch.float32).flatten()
        output_grid = torch.tensor(data['output'], dtype=torch.float32).flatten()
        return input_grid, output_grid

# --- 3. Training and Evaluation ---

def vae_loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (input_data, output_data) in enumerate(dataloader):
        input_data = input_data.to(device)
        output_data = output_data.to(device)
        
        optimizer.zero_grad()
        
        # For VAE training, the target is a shifted version of the output
        # to enable teacher forcing in the autoregressive decoder.
        # The decoder at step `i` should predict token `i`, given tokens `0..i-1`.
        # We prepend a "zero" token to the start.
        decoder_input = torch.cat([torch.zeros(output_data.shape[0], 1).to(device), output_data[:, :-1]], dim=1)

        recon_batch, mu, log_var = model(input_data, decoder_input)
        
        loss = vae_loss_function(recon_batch, output_data, mu, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader.dataset)

def visualize_results(model, dataloader, device, filename='inference_results.png'):
    model.eval()
    inputs, outputs = next(iter(dataloader))
    inputs, outputs = inputs.to(device), outputs.to(device)

    with torch.no_grad():
        generated_outputs = model.generate(inputs)

    inputs = inputs.cpu().numpy().reshape(-1, 10, 10)
    outputs = outputs.cpu().numpy().reshape(-1, 10, 10)
    generated_outputs = generated_outputs.cpu().numpy()

    n_samples = min(inputs.shape[0], 5)
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, n_samples * 3))
    fig.suptitle('Model Inference Results', fontsize=16)
    
    for i in range(n_samples):
        axes[i, 0].imshow(inputs[i], cmap='Greys', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Input-{i}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(outputs[i], cmap='viridis', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Ground Truth-{i}')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(generated_outputs[i], cmap='viridis', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Generated-{i}')
        axes[i, 2].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    print(f"Inference visualization saved to {filename}")


if __name__ == '__main__':
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = 'data'
    NUM_EPOCHS = 2000 # Increased for better convergence
    LEARNING_RATE = 0.001
    BATCH_SIZE = 8

    print(f"Using device: {DEVICE}")

    # --- Data Loaders ---
    train_dataset = GridDataset(os.path.join(DATA_DIR, 'train'))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    test_dataset = GridDataset(os.path.join(DATA_DIR, 'test'))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model Initialization ---
    model = TransformerVAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train(model, train_loader, optimizer, DEVICE)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {loss:.4f}')
    print("Training finished.")

    # --- Save the model ---
    MODEL_SAVE_PATH = 'vae_transformer.pth'
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # --- Visualization ---
    visualize_results(model, test_loader, DEVICE)
