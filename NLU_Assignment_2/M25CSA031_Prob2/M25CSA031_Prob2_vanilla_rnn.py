"""
Problem 2: Vanilla RNN

Name generation using Vanilla RNN.
Implementing through manual weight matrices and tanh.


Architecture overview:
    Embedding layer which converting charecter to vector
    Hidden layer which processing the input and previous hidden state
    Output linear layer which projects hidden state to vocabulary logits

    Name generating by sampling from softmax at each timestep

Hyperparameters:
    Hidden size: 128
    Embedding dim: 64
    Learning rate: 0.005
    Epochs: 25
    Batch size: 64
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

# Data loading and vocabulary


def load_training_names(filepath="TrainingNames.txt"):
    """Read names from training file."""
    with open(filepath, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def build_character_vocab(names):
    """
    for vocab creation building char to index ad index to char mappings.

    3 special tokens:
      PAD for padding shorter sequences in a batch
      SOS start of sequense
      EOS end of sequense
    """
    unique_chars = sorted(set("".join(names)))

    char_to_idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    for ch in unique_chars:
        if ch not in char_to_idx:
            char_to_idx[ch] = len(char_to_idx)

    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char


class NameDataset(torch.utils.data.Dataset):
    """
    for dataset creation converting each name string into a tensor of character indices.
    For example: "Raj" -> [SOS, R, a, j, EOS] -> [1, 34, 5, 18, 2]


    """
    def __init__(self, names, char_to_idx):
        self.sequences = []
        for name in names:
            seq = [char_to_idx["<SOS>"]] + [char_to_idx[c] for c in name] + [char_to_idx["<EOS>"]]
            self.sequences.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def pad_batch(batch):
    """Pad all sequences in a batch to the same length."""
    max_len = max(len(seq) for seq in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


# Vanilla RNN Model

class VanillaRNN(nn.Module):
    """
    rnn model using weigths and tanh.


    
    RNN : h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
    
    Where:
        x_t     = input embedding at timestep t
        h_{t-1} = hidden state from previous timestep
        W_xh    = input to hidden weight matrix
        W_hh    = hidden to hidden weight matrix 
        b_h     = bias term
    
    Output at every timestep is a linear projection of h_t to vocab size.

    """
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Character embedding which maps each character index to a dense vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Manual RNN weights

        # W_xh transforms the input embedding into hidden space
        self.W_xh = nn.Parameter(torch.empty(embed_dim, hidden_size))

        # W_hh transforms the previous hidden state which creates the recurrence
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        # Bias term for the hidden layer
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # Output projection which maps hidden state back to vocabulary.
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        # Initializing weights using Xavier for stable gradients
        self._init_weights()

    def _init_weights(self):
        """For preventing vanishing/exploding gradients at the start of training xavior weights are used."""
        nn.init.xavier_uniform_(self.W_xh)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Initializing hidden state to zeros at the start of each sequence
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []

        # Process one timestep at a time
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # current input

            # Core RNN equation: h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b)
            h = torch.tanh(
                torch.matmul(x_t, self.W_xh) +
                torch.matmul(h, self.W_hh) +
                self.b_h
            )
            outputs.append(h.unsqueeze(1))

        # Stack all hidden states along the time dimension
        all_hidden = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_size)
        logits = self.output_layer(all_hidden)   # (batch, seq_len, vocab_size)
        return logits


# Training

def train_model(model, dataloader, epochs, lr, device):
    """
    training loop with cross-entropy loss.
    
    Model is learns to predict a next charecter using all previous charecters.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(device)
            # Input everything except the last character
            inp = batch[:, :-1]
            # Target everything except the first character
            target = batch[:, 1:]

            predictions = model(inp)
            loss = loss_fn(predictions.reshape(-1, predictions.size(-1)), target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()

            # Clipping gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / num_batches
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.5f}")

    return model


# Name Generation

def generate_name(model, char_to_idx, idx_to_char, device, max_len=25, temperature=1.4):
    """
    generating name using the trained model charecter by charecter.
    

    Process:
    Starting with the <SOS> token, feeding it into RNN to get the next character distribution
    Sample from that distribution, repeat until <EOS> is generated or max_len is reached
    
    Temperature controls randomness:
    Low temperature (0.5) = very conservative, common names
    High temperature (1.5) = more creative, novel but possibly weird names

    """
    model.eval()
    with torch.no_grad():
        h = torch.zeros(1, model.hidden_size, device=device)
        current_char = torch.tensor([[char_to_idx["<SOS>"]]], device=device)
        generated_chars = []

        for _ in range(max_len):
            emb = model.embedding(current_char)[:, 0, :]
            h = torch.tanh(
                torch.matmul(emb, model.W_xh) +
                torch.matmul(h, model.W_hh) +
                model.b_h
            )

            logits = model.output_layer(h) / temperature
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()

            if next_idx == char_to_idx["<EOS>"] or next_idx == char_to_idx["<PAD>"]:
                break

            generated_chars.append(idx_to_char[next_idx])
            current_char = torch.tensor([[next_idx]], device=device)

    return "".join(generated_chars).strip()


# Main

if __name__ == "__main__":
    # Hyperparameters
    HIDDEN_SIZE = 128
    EMBED_DIM = 64
    LEARNING_RATE = 0.005
    EPOCHS = 25
    BATCH_SIZE = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Loading data
    names = load_training_names()
    char_to_idx, idx_to_char = build_character_vocab(names)
    vocab_size = len(char_to_idx)
    print(f"Loaded {len(names)} training names, vocabulary size = {vocab_size}")

    # Creating dataset and dataloader
    dataset = NameDataset(names, char_to_idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_batch)

    # model building
    model = VanillaRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*50}")
    print(f"  VANILLA RNN")
    print(f"  Trainable parameters: {num_params:,}")
    print(f"  Hidden size: {HIDDEN_SIZE}, Embed dim: {EMBED_DIM}")
    print(f"  LR: {LEARNING_RATE}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")
    print(f"{'='*50}")

    # Training
    print("\nTraining...")
    model = train_model(model, loader, EPOCHS, LEARNING_RATE, device)

    # Save model to models/ folder
    os.makedirs("models", exist_ok=True)
    save_path = "models/M25CSA031_Prob2_vanilla_rnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Generate sample names
    print("\nSample Generated Names:")
    for i in range(15):
        name = generate_name(model, char_to_idx, idx_to_char, device, temperature=1.4)
        print(f"  {i+1}. {name}")

    # Save 200 generated names for evaluation
    output_path = "generated_vanilla_rnn.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(200):
            f.write(generate_name(model, char_to_idx, idx_to_char, device, temperature=1.5) + "\n")
    print(f"\nSaved 200 generated names to {output_path}")
