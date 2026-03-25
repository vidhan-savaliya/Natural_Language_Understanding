# Problem 2: Character-Level Name Generation

## Prerequisites

```bash
pip install torch numpy
```

## File Structure

| File                               | Description                                          |
| ---------------------------------- | ---------------------------------------------------- |
| `TrainingNames.txt`                | 1000 Indian names dataset                            |
| `M25CSA031_Prob2_vanilla_rnn.py`   | Vanilla RNN (from scratch, manual weight matrices)   |
| `M25CSA031_Prob2_blstm.py`         | Bidirectional LSTM (from scratch, manual LSTM gates) |
| `M25CSA031_Prob2_rnn_attention.py` | RNN + Bahdanau Attention (from scratch)              |
| `M25CSA031_Prob2_evaluation.py`    | Quantitative metrics (Novelty Rate + Diversity)      |
| `M25CSA031_Prob2_analysis.py`      | Qualitative analysis (realism + failure modes)       |

## How to Run

Run the scripts **in order** from the `M25CSA031_Prob2/` directory:

```bash
cd M25CSA031_Prob2

# Step 1: Train Vanilla RNN and generate 200 names
python M25CSA031_Prob2_vanilla_rnn.py

# Step 2: Train BLSTM and generate 200 names
python M25CSA031_Prob2_blstm.py

# Step 3: Train RNN + Attention and generate 200 names
python M25CSA031_Prob2_rnn_attention.py

# Step 4: Compute Novelty Rate and Diversity for all models
python M25CSA031_Prob2_evaluation.py

# Step 5: Run qualitative analysis (realism checks, failure modes)
python M25CSA031_Prob2_analysis.py
```

## Output Files

| Output          | Location                                   |
| --------------- | ------------------------------------------ |
| Trained models  | `models/M25CSA031_Prob2_vanilla_rnn.pth`   |
|                 | `models/M25CSA031_Prob2_blstm.pth`         |
|                 | `models/M25CSA031_Prob2_rnn_attention.pth` |
| Generated names | `generated_vanilla_rnn.txt`                |
|                 | `generated_blstm.txt`                      |
|                 | `generated_rnn_attention.txt`              |

## Notes

- GPU is used automatically if available (`cuda`), otherwise falls back to CPU.
- BLSTM training takes the longest (~5 min on GPU, ~20 min on CPU) due to 50 epochs and stacked layers.
- Vanilla RNN and RNN+Attention train in ~1-2 minutes each.
- evaluation and analysis require the generated text files from Steps 1-3, so run the models first.
- All results are printed to the console.
