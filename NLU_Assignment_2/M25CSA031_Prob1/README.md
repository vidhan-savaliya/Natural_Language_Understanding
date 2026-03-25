# Problem 1: Learning Word Embeddings from IIT Jodhpur Data

## Prerequisites

```bash
pip install numpy nltk gensim matplotlib wordcloud scikit-learn requests beautifulsoup4 PyPDF2
```

## File Structure

| File                                   | Description                                          |
| -------------------------------------- | ---------------------------------------------------- |
| `M25CSA031_prob1_scraper.py`           | BFS crawler for IIT Jodhpur website (HTML + PDF)     |
| `M25CSA031_prob1_data_preparation.py`  | Text cleaning, preprocessing, word cloud             |
| `M25CSA031_prob1_word2vec_scratch.py`  | Word2Vec from scratch (NumPy only, CBOW + Skip-gram) |
| `M25CSA031_prob1_training.py`          | Trains both scratch and Gensim models, compares them |
| `M25CSA031_prob1_semantic_analysis.py` | Nearest neighbors + word analogies on all models     |
| `M25CSA031_prob1_visualization.py`     | PCA and t-SNE plots of word embeddings               |

## How to Run

Run the scripts **in order** from the `M25CSA031_Prob1/` directory:

```bash
cd M25CSA031_Prob1

# Step 1: Scrape data from IIT Jodhpur website (skip if raw_corpus.txt already exists)
python M25CSA031_prob1_scraper.py

# Step 2: Clean and preprocess the corpus
python M25CSA031_prob1_data_preparation.py

# Step 3: Train Word2Vec models (scratch + Gensim)
python M25CSA031_prob1_training.py

# Step 4: Run semantic analysis (nearest neighbors + analogies)
python M25CSA031_prob1_semantic_analysis.py

# Step 5: Generate visualization plots
python M25CSA031_prob1_visualization.py
```

## Output Files

| Output              | Location                                                         |
| ------------------- | ---------------------------------------------------------------- |
| Cleaned corpus      | `clened_corpus.txt`                                              |
| Word cloud image    | `M25CSA031_prob1_wordcloud.png`                                  |
| Gensim models       | `M25CSA031_gensim_cbow.model`, `M25CSA031_gensim_skipgram.model` |
| Scratch models      | `M25CSA031_scratch_cbow.pkl`, `M25CSA031_scratch_skipgram.pkl`   |
| Visualization plots | `results/` folder                                                |

## Notes

- Step 1 (scraper) takes ~10-15 minutes depending on network speed. You can skip it because `raw_corpus.txt` already exists.
- Step 3 (training) takes ~5-10 minutes for scratch models on CPU.
- All results are printed to the console.
