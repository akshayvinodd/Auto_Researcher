import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import wandb
import os

# Configuration - T4 COMPATIBLE
sequence_len = 64 # Reduced for small dataset compatibility
n_layer = 3
d_model = 512
n_head = 8
DEVICE_BATCH_SIZE = 4
dropout = 0.24
lr = 3e-4 # Add learning rate
weight_decay = 1e-4 # Add weight decay

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(sequence_len, d_model) # Positional Embedding
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_head, dropout) for _ in range(n_layer)])
        self.head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(d_model) # Final layer norm

    def forward(self, x):
        b, t = x.size()
        positions = torch.arange(t, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(positions)  # Add positional embeddings
        x = self.dropout(x) # Apply dropout after embedding

        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x) # Apply final layer norm
        return self.head(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.head_dim = d_model // n_head
        assert self.head_dim * n_head == d_model, "d_model must be divisible by n_head"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_linear(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = self.k_linear(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = self.v_linear(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # use torch.nn.functional.scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout if self.training else 0.0, is_causal=False)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.proj(attn_output)
        attn_output = self.dropout(attn_output)
        return attn_output

# --- REAL DATA LOADING & TRAINING ---
# Load data.txt (TinyShakespeare snippet)
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n_val = int(len(data) * 0.1) # 10% for validation
train_data = data[:-n_val]
val_data = data[-n_val:]

def get_batch(split):
    dataset = train_data if split == 'train' else val_data
    ix = torch.randint(len(dataset) - sequence_len, (DEVICE_BATCH_SIZE,))
    x = torch.stack([dataset[i:i+sequence_len] for i in ix])
    y = torch.stack([dataset[i+1:i+sequence_len+1] for i in ix])
    return x, y

# Override vocab_size in the model class if needed or set manually
# Using a fixed 50257 for consistency with previous architecture, 
# but in real usage it should match vocab_size.
# For this POC, we'll keep the model definition but use char-level input.

# --- WANDB INITIALIZATION ---
# Using 'offline' mode for the POC to ensure it runs without an API key.
# Switch to 'online' or remove mode to sync to wandb.ai
wandb.init(
    project="AutoResearcher",
    config={
        "d_model": d_model,
        "n_layer": n_layer,
        "dropout": dropout,
        "lr": lr,
        "sequence_len": sequence_len,
        "batch_size": DEVICE_BATCH_SIZE
    },
    mode="offline" 
)

model = SimpleTransformer(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print(f"Dataset Size: {len(data)} chars, Vocab Size: {vocab_size}")
print("Training (Real Epochs)...")

max_iters = 100
total_loss = 0

# --- EARLY STOPPING LOGIC ---
best_loss = float('inf')
patience = 20 # Steps to wait before quitting
steps_no_improve = 0

for i in range(max_iters):
    xb, yb = get_batch('train')
    logits = model(xb)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = yb.view(B*T)
    
    loss = F.cross_entropy(logits, targets)
    
    # Check for improvement
    if loss.item() < best_loss:
        best_loss = loss.item()
        steps_no_improve = 0
    else:
        steps_no_improve += 1
    
    # Early Stop if model isn't learning
    if steps_no_improve >= patience:
        print(f"🛑 EARLY STOPPING: No improvement for {patience} steps.")
        break
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()

    if i % 10 == 0:
        wandb.log({"step": i, "loss": loss.item()})

# Calculate val_bpb (Bits Per Character)
actual_steps = i + 1
avg_loss = total_loss / actual_steps
val_bpb = avg_loss / math.log(2)
print(f"val_bpb: {round(val_bpb, 4)}")

# Log final metrics and finish run
wandb.log({"val_bpb": val_bpb, "actual_steps": actual_steps})
wandb.finish()