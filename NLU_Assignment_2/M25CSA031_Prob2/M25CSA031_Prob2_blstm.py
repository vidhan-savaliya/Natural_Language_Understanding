"""
Problem 2 : Bidirectional LSTM

Name generation using BiLSTM.
LSTM gates is implemented using raw weight matrices.


Architecture overview:
    Embedding layer
    Forward LSTM cell for left to right
    Backward LSTM cell for right to left
    Hidden states, which are concatenated at each timestep
    Stack a Layers for residual connections
    seperate generation head for forward pass for inference
    Layer normalization after each BiLSTM layer for training stability

LSTM gate equations (per cell):
    i_t = sigmoid(x_t @ W_xi + h_{t-1} @ W_hi + b_i)   — input gate
    f_t = sigmoid(x_t @ W_xf + h_{t-1} @ W_hf + b_f)   — forget gate
    g_t = tanh(x_t @ W_xg + h_{t-1} @ W_hg + b_g)      — cell candidate
    o_t = sigmoid(x_t @ W_xo + h_{t-1} @ W_ho + b_o)    — output gate
    c_t = f_t * c_{t-1} + i_t * g_t                      — cell state update
    h_t = o_t * tanh(c_t)                                 — hidden state

Hyperparameters:
    Hidden size: 128
    Embedding dim: 64
    Stacked layers: 2
    Dropout: 0.3
    Learning rate: 0.003
    Epochs: 50
    Batch size: 64
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os


# Data loading and vocabulary

def load_training_names(filepath="TrainingNames.txt"):
    """Read names from training file."""
    with open(filepath, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names

# building charecter vocabulary
def build_character_vocab(names):
    """Build char to index mappings with PAD, SOS, EOS special tokens."""
    chars = sorted(set("".join(names)))
    stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    for ch in chars:
        stoi[ch] = len(stoi)
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


class NameDataset(Dataset):
    """Encodes each name as SOS, c1, c2, ....cn, EOS tensor."""
    def __init__(self, names, stoi):
        self.data = []
        for name in names:
            encoded = [stoi["<SOS>"]] + [stoi[ch] for ch in name] + [stoi["<EOS>"]]
            self.data.append(torch.tensor(encoded, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad_batch(batch):
    """Padding for same length."""
    max_len = max(s.size(0) for s in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(batch):
        padded[i, :s.size(0)] = s
    return padded


# LSTM Cell

class LSTMCell(nn.Module):
    """
    A single LSTM cell with all 4 gates implemented.
    
    Combining all gates weights matrics into big matrix for efficiency.
    gates = x @ W_x + h @ W_h + bias
    after that splitting the results in 4 equal chunks for i, f, g, o gates.
    
    Forgot gate bias is initialized to 1.0 for better gradient flow.
    thats why the model doesn't forget everything at the start of training.

    """
    def __init__(self, input_size, hidden_size, use_layer_norm=True):
        super().__init__()
        self.hidden_size = hidden_size

        # Combined weight matrices for all 4 gates
        self.W_x = nn.Parameter(torch.randn(input_size, 4 * hidden_size) * 0.01)
        self.W_h = nn.Parameter(torch.randn(hidden_size, 4 * hidden_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

        # For better gradient flow initialize forget gate bias to 1.0
        with torch.no_grad():
            self.bias[hidden_size:2 * hidden_size].fill_(1.0)

        # For training stability apply layer norm on hidden output
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else None

    def forward(self, x_t, h_prev, c_prev):
        """
        One timestep of the LSTM.
        x_t    : (batch, input_size)  — current input
        h_prev : (batch, hidden_size) — previous hidden state
        c_prev : (batch, hidden_size) — previous cell state
        Returns: (h_new, c_new)
        """
        # Compute all 4 gates at once
        gates = x_t @ self.W_x + h_prev @ self.W_h + self.bias

        # Splitting into individual gates
        i_gate = torch.sigmoid(gates[:, :self.hidden_size])                    # input gate
        f_gate = torch.sigmoid(gates[:, self.hidden_size:2*self.hidden_size])  # forget gate
        g_gate = torch.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])   # cell candidate
        o_gate = torch.sigmoid(gates[:, 3*self.hidden_size:])                  # output gate

        # Cell state : forget old info + add new info
        c_new = f_gate * c_prev + i_gate * g_gate

        # Hidden state: filtered version of cell state
        h_new = o_gate * torch.tanh(c_new)

        if self.layer_norm is not None:
            h_new = self.layer_norm(h_new)

        return h_new, c_new


# Bidirectional LSTM Model


class BiLSTM(nn.Module):
    """
    Bi LSTM with stacked layers and a dual head architecture.
    
    In training uses both forward and backward cells for outputs through fc_out
    In generation uses only the forward cell path + gen_cells, outputs through fc_gen
    
    Two heads because during time of generation we can go in one direction, so we ned to seperate trained head that works with forward only hidden states.

    Both heads are trained simultaneously so fc_gen gets proper gradients.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Layer 0 : forward and backward LSTM cells
        self.cell_fw = LSTMCell(embed_dim, hidden_size, use_layer_norm=True)
        self.cell_bw = LSTMCell(embed_dim, hidden_size, use_layer_norm=True)

        # Additional stacked layers in that input is 2*hidden from concatenation
        self.fw_cells = nn.ModuleList([
            LSTMCell(2 * hidden_size, hidden_size, use_layer_norm=True)
            for _ in range(num_layers - 1)
        ])
        self.bw_cells = nn.ModuleList([
            LSTMCell(2 * hidden_size, hidden_size, use_layer_norm=True)
            for _ in range(num_layers - 1)
        ])

        # LayerNorm after each stacked layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(2 * hidden_size) for _ in range(num_layers)
        ])

        # Dropout between stacked layers
        self.dropout = nn.Dropout(dropout)

        # Training head : bidirectional output
        self.fc_out = nn.Linear(2 * hidden_size, vocab_size)

        # Generation head: forward-only output
        self.fc_gen = nn.Linear(hidden_size, vocab_size)

        # Forward-only stacked cells for generation
        self.gen_cells = nn.ModuleList([
            LSTMCell(hidden_size, hidden_size, use_layer_norm=True)
            for _ in range(num_layers - 1)
        ])

    def forward(self, x):
        batch_size, seq_len = x.shape
        emb = self.embed(x)

        # Layer 0 : Bidirectional pass
        h_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
        fw_outs = []
        for t in range(seq_len):
            h_f, c_f = self.cell_fw(emb[:, t, :], h_f, c_f)
            fw_outs.append(h_f)

        h_b = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_b = torch.zeros(batch_size, self.hidden_size, device=x.device)
        bw_outs = [None] * seq_len
        for t in range(seq_len - 1, -1, -1):
            h_b, c_b = self.cell_bw(emb[:, t, :], h_b, c_b)
            bw_outs[t] = h_b

        # Concatenate forward + backward
        combined = torch.cat([
            torch.cat([fw_outs[t], bw_outs[t]], dim=1).unsqueeze(1)
            for t in range(seq_len)
        ], dim=1)
        combined = self.layer_norms[0](combined)

        # Stacked layers with residual connections
        for layer_idx in range(self.num_layers - 1):
            layer_in = self.dropout(combined)

            h_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
            fw_outs = []
            for t in range(seq_len):
                h_f, c_f = self.fw_cells[layer_idx](layer_in[:, t, :], h_f, c_f)
                fw_outs.append(h_f)

            h_b = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_b = torch.zeros(batch_size, self.hidden_size, device=x.device)
            bw_outs = [None] * seq_len
            for t in range(seq_len - 1, -1, -1):
                h_b, c_b = self.bw_cells[layer_idx](layer_in[:, t, :], h_b, c_b)
                bw_outs[t] = h_b

            stacked = torch.cat([
                torch.cat([fw_outs[t], bw_outs[t]], dim=1).unsqueeze(1)
                for t in range(seq_len)
            ], dim=1)
            stacked = self.layer_norms[layer_idx + 1](stacked)
            combined = stacked + layer_in  # residual connection

        # BiLSTM training head
        logits = self.fc_out(combined)

        # Generation head forward
        h_g = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_g = torch.zeros(batch_size, self.hidden_size, device=x.device)
        gen_outs = []
        for t in range(seq_len):
            h_g, c_g = self.cell_fw(emb[:, t, :], h_g, c_g)
            gen_outs.append(h_g)

        gen_hidden = torch.stack(gen_outs, dim=1)
        for cell in self.gen_cells:
            h_s = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_s = torch.zeros(batch_size, self.hidden_size, device=x.device)
            layer_outs = []
            for t in range(seq_len):
                h_s, c_s = cell(gen_hidden[:, t, :], h_s, c_s)
                layer_outs.append(h_s)
            layer_out = torch.stack(layer_outs, dim=1)
            gen_hidden = layer_out + gen_hidden  # residual

        logits_gen = self.fc_gen(gen_hidden)

        return logits, logits_gen


# Training

def train_blstm(model, dataloader, epochs, lr, device):
    """

    Taining with dual loss from BiLSTM head and generation head.
    Both losses ensure that the generation path learns properly.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0

        for batch in dataloader:
            batch = batch.to(device)
            inp = batch[:, :-1]
            target = batch[:, 1:]

            logits, logits_gen = model(inp)

            # Combined loss from both heads
            loss_bi = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss_gen = loss_fn(logits_gen.reshape(-1, logits_gen.size(-1)), target.reshape(-1))
            loss = loss_bi + loss_gen

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}  Loss: {total_loss/count:.4f}")

    return model


# Name Generation

def generate_name(model, stoi, itos, device, max_len=30, temperature=0.8):
    """
    Name generation using forward path and generation head.
    Backward cell is not using during generation because dont have a future charecters
    generation is left to right.
    """
    model.eval()
    with torch.no_grad():
        hs = [torch.zeros(1, model.hidden_size, device=device) for _ in range(model.num_layers)]
        cs = [torch.zeros(1, model.hidden_size, device=device) for _ in range(model.num_layers)]
        inp_idx = stoi["<SOS>"]
        result = []

        for _ in range(max_len):
            inp_tensor = torch.tensor([[inp_idx]], device=device)
            emb = model.embed(inp_tensor)[:, 0, :]

            # Layer 0 : using shared forward cell
            hs[0], cs[0] = model.cell_fw(emb, hs[0], cs[0])
            out = hs[0]

            # Stacked layers through gen_cells with residual
            for layer_idx in range(model.num_layers - 1):
                skip = out
                hs[layer_idx + 1], cs[layer_idx + 1] = model.gen_cells[layer_idx](
                    out, hs[layer_idx + 1], cs[layer_idx + 1]
                )
                out = hs[layer_idx + 1] + skip

            logits = model.fc_gen(out) / temperature
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()

            if next_idx in (stoi["<EOS>"], stoi["<PAD>"], stoi["<SOS>"]):
                break
            result.append(itos[next_idx])
            inp_idx = next_idx

    return "".join(result)


# Main execution

if __name__ == "__main__":
    HIDDEN_SIZE = 128
    EMBED_DIM = 64
    LR = 0.003
    EPOCHS = 50
    BATCH_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    names = load_training_names()
    stoi, itos = build_character_vocab(names)
    vocab_size = len(stoi)
    print(f"Loaded {len(names)} names, vocab size = {vocab_size}")

    dataset = NameDataset(names, stoi)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_batch)

    model = BiLSTM(vocab_size, EMBED_DIM, HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*50}")
    print(f"  BIDIRECTIONAL LSTM (From Scratch)")
    print(f"  Trainable parameters: {num_params:,}")
    print(f"  Hidden: {HIDDEN_SIZE}, Layers: {NUM_LAYERS}, Embed: {EMBED_DIM}")
    print(f"  LR: {LR}, Epochs: {EPOCHS}, Dropout: {DROPOUT}")
    print(f"{'='*50}")

    print(f"\nTraining for {EPOCHS} epochs...")
    model = train_blstm(model, loader, EPOCHS, LR, device)

    os.makedirs("models", exist_ok=True)
    save_path = "models/M25CSA031_Prob2_blstm.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    print("\nSample Generated Names:")
    for i in range(20):
        name = generate_name(model, stoi, itos, device)
        print(f"  {i+1}. {name}")

    with open("generated_blstm.txt", "w", encoding="utf-8") as f:
        for _ in range(200):
            f.write(generate_name(model, stoi, itos, device) + "\n")
    print("\nSaved 200 generated names to generated_blstm.txt")
