"""
Problem 2 : RNN with Attention

Name generation using from scratch RNN + Additive Attention.


Architecture overview:
    Embedding layer
    manual rnn cell same as vanilla rnn
    additive attention mechanism over all past hidden states
    computation of context vector as weighted sum of past hiddens
    output projection from concatenated [h_t, context] to vocabulary



Attention equations:
    energy_t, j = v^T * tanh(W_a @ h_t + U_a @ h_j)   for each past position j
    alpha_t, j  = softmax(energy_t, j)                attention weights
    context_t  = sum_j(alpha_t, j * h_j)              weighted combination


without attention the rnn must compress all previous character information into a single fixed size hidden vector
with attention the model can directly look back at specific earlier characters when deciding the next one
this is especially useful for maintaining consistency in longer names

Hyperparameters:
    Hidden size: 128
    Embedding dim: 64
    Dropout: 0.2
    Learning rate: 0.005
    Epochs: 30
    Batch size: 64
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os


# Data loading and vocabulary


def load_training_names(filepath="TrainingNames.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names

# Vocabulary building
def build_character_vocab(names):
    chars = sorted(set("".join(names)))
    stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    for ch in chars:
        stoi[ch] = len(stoi)
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


# Dataset class
class NameDataset(Dataset):
    def __init__(self, names, stoi):
        self.data = []
        for name in names:
            encoded = [stoi["<SOS>"]] + [stoi[ch] for ch in name] + [stoi["<EOS>"]]
            self.data.append(torch.tensor(encoded, dtype=torch.long))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# Padding function
def pad_batch(batch):
    max_len = max(len(s) for s in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(batch):
        padded[i, :len(s)] = s
    return padded


# Manual Dropout

def apply_dropout(x, p=0.2, training=True):
    """
    Dropout implementation.
    During training randomly doing zeroing out elements with probability p,
    after that scale up remaining values by 1/(1-p) to maintain expected sum.
    """
    if not training or p == 0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return mask * x / (1.0 - p)


# RNN + Attention Model

class RNNWithAttention(nn.Module):
    """
    RNN with additive (Bahdanau) attention mechanism.
    
    at every timestep after computing hidden state also computing attention weighted context vector
    over all previous hidden state.

    The final output combines both the current hidden state and this context.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, dropout_p=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Manual RNN weights (same as Vanilla RNN)
        self.W_xh = nn.Parameter(torch.empty(embed_dim, hidden_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # Attention weights (Bahdanau-style additive attention)
        # W_a projects current hidden state
        self.W_a = nn.Parameter(torch.empty(hidden_size, hidden_size))
        # U_a projects past hidden states
        self.U_a = nn.Parameter(torch.empty(hidden_size, hidden_size))
        # v_a computes scalar energy from the combined projection
        self.v_a = nn.Parameter(torch.empty(hidden_size, 1))

        # Output: takes concatenated [hidden, context] = 2*hidden_size → vocab
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all weight matrices."""
        nn.init.xavier_uniform_(self.W_xh)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.xavier_uniform_(self.W_a)
        nn.init.xavier_uniform_(self.U_a)
        nn.init.xavier_uniform_(self.v_a)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def compute_attention(self, h_current, past_hiddens):
        """

        Computing attention weights and context vector.
        
        h_current:    (batch, hidden) current timestep's hidden state
        past_hiddens: (batch, t, hidden) all hidden states up to now
        

        Returns: context vector (batch, hidden)

        """
        batch_size, t, _ = past_hiddens.shape

        # For matching past states shape expanding current hidden states.

        h_expanded = h_current.unsqueeze(1).expand(-1, t, -1)

        # Additive attention : energy = v^T * tanh(W*h_current + U*h_past)
        energy = torch.tanh((h_expanded @ self.W_a) + (past_hiddens @ self.U_a))
        energy = energy @ self.v_a  # (batch, t, 1)

        # For getting attention weights do a softmax over time dimension
        weights = torch.softmax(energy, dim=1)  # (batch, t, 1)

        # Context is weighted sum of past hidden states
        context = (weights * past_hiddens).sum(dim=1)  # (batch, hidden)
        return context

    def forward(self, x):
        batch_size, seq_len = x.shape
        emb = self.embed(x)
        emb = apply_dropout(emb, self.dropout_p, self.training)

        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        all_outputs = []
        past_hiddens_list = []

        for t in range(seq_len):
            x_t = emb[:, t, :]

            # RNN step
            h = torch.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)
            past_hiddens_list.append(h.unsqueeze(1))

            # Attention over all past hidden states 
            past_hiddens = torch.cat(past_hiddens_list, dim=1)
            context = self.compute_attention(h, past_hiddens)

            # Combining hidden state with context
            combined = torch.cat([h, context], dim=1)
            all_outputs.append(combined.unsqueeze(1))

        out = torch.cat(all_outputs, dim=1)
        out = apply_dropout(out, self.dropout_p, self.training)
        logits = self.fc_out(out)
        return logits


# Training

def train_model(model, dataloader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0

        for batch in dataloader:
            batch = batch.to(device)
            inp = batch[:, :-1]
            target = batch[:, 1:]

            logits = model(inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            count += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:2d}/{epochs}  Loss: {total_loss/count:.4f}  LR: {current_lr:.6f}")

    return model


# Name Generation

def generate_name(model, stoi, itos, device, max_len=30, temperature=0.8):
    model.eval()
    with torch.no_grad():
        h = torch.zeros(1, model.hidden_size, device=device)
        inp_idx = stoi["<SOS>"]
        result = []
        past_hiddens_list = []

        for _ in range(max_len):
            inp_tensor = torch.tensor([[inp_idx]], device=device)
            emb = model.embed(inp_tensor)[:, 0, :]

            h = torch.tanh(emb @ model.W_xh + h @ model.W_hh + model.b_h)
            past_hiddens_list.append(h.unsqueeze(1))

            past_hiddens = torch.cat(past_hiddens_list, dim=1)
            context = model.compute_attention(h, past_hiddens)

            combined = torch.cat([h, context], dim=1)
            logits = model.fc_out(combined) / temperature
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()

            if next_idx == stoi["<EOS>"] or next_idx == stoi["<PAD>"]:
                break
            result.append(itos[next_idx])
            inp_idx = next_idx

    return "".join(result)


# Main execution


if __name__ == "__main__":
    HIDDEN_SIZE = 128
    EMBED_DIM = 64
    LR = 0.005
    EPOCHS = 30
    DROPOUT = 0.2
    BATCH_SIZE = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    names = load_training_names()
    stoi, itos = build_character_vocab(names)
    vocab_size = len(stoi)
    print(f"Loaded {len(names)} names, vocab size = {vocab_size}")

    dataset = NameDataset(names, stoi)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_batch)

    model = RNNWithAttention(vocab_size, EMBED_DIM, HIDDEN_SIZE, DROPOUT)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*50}")
    print(f"  RNN WITH ATTENTION (From Scratch)")
    print(f"  Trainable parameters: {num_params:,}")
    print(f"  Hidden: {HIDDEN_SIZE}, Embed: {EMBED_DIM}, Dropout: {DROPOUT}")
    print(f"  LR: {LR}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")
    print(f"{'='*50}")

    print(f"\nTraining for {EPOCHS} epochs...")
    model = train_model(model, loader, EPOCHS, LR, device)

    os.makedirs("models", exist_ok=True)
    save_path = "models/M25CSA031_Prob2_rnn_attention.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    print("\nSample Generated Names:")
    for i in range(20):
        name = generate_name(model, stoi, itos, device, temperature=1.2)
        print(f"  {i+1}. {name}")

    with open("generated_rnn_attention.txt", "w", encoding="utf-8") as f:
        for _ in range(200):
            f.write(generate_name(model, stoi, itos, device, temperature=1.2) + "\n")
    print("\nSaved 200 generated names to generated_rnn_attention.txt")
