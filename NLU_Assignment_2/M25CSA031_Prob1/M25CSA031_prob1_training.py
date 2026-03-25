"""
Problem 1 : Model Training

Training both CBOW and Skip-gram models using two approaches from scratch and Using Gensim

Different Hyperparameters i used for experiments:
  - Embedding dimension: 50, 100
  - Context window size: 3, 5
  - Negative samples: 5, 10

"""

import time
import os
import pickle
from gensim.models import Word2Vec as GensimWord2Vec

# importing from scratch implementation
from M25CSA031_prob1_word2vec_scratch import ScratchWord2Vec


def corpus_loader(filepath):
    """
    this is readinng a cleaned corpusfile and each lines and each sentence is split into a list of words,
    becuase both Gensim and scratch model expect this format

    """
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            if len(words) >= 2:  # need at least 2 words for context
                sentences.append(words)
    
    print(f"Loaded {len(sentences)} sentences for training")
    return sentences


def training_gensim_models(sentences):
    """
    Training of gensim models with different hyperparameters

    Gensim is well optimized implementation so it is faster than numpy scratch version, using it for comparison.

    sg=0 means CBOW, sg=1 means Skip-gram in Gensim's API.
    """
    dimensions = [50, 100]
    windows = [3, 5]
    neg_samples = [5, 10]

    print("\n" + "=" * 65)
    print("  GENSIM MODEL TRAINING:  HYPERPARAMETER EXPERIMENTS")
    print("=" * 65)

    gensim_models = {}

    for model_type, sg_flag in [("CBOW", 0), ("Skip-gram", 1)]:
        print(f"\n  {model_type} :")
        
        for dim in dimensions:
            for win in windows:
                for neg in neg_samples:
                    label = f"{model_type}_d{dim}_w{win}_n{neg}"
                    
                    print(f"  Training {label} ... ", end="", flush=True)
                    t0 = time.time()
                    
                    model = GensimWord2Vec(
                        sentences=sentences,
                        vector_size=dim,
                        window=win,
                        negative=neg,
                        sg=sg_flag,
                        min_count=2,
                        workers=4,
                        epochs=10
                    )
                    
                    elapsed = time.time() - t0
                    vocab_sz = len(model.wv)
                    print(f"Done in {elapsed:.1f}s (vocab: {vocab_sz})")
                    
                    gensim_models[label] = model

    # Save the baseline models (dim=100, win=5, neg=5) for semantic analysis
    best_cbow = gensim_models.get("CBOW_d100_w5_n5")
    best_sg = gensim_models.get("Skip-gram_d100_w5_n5")
    
    if best_cbow:
        best_cbow.save("M25CSA031_gensim_cbow.model")
        print(f"\n  Saved baseline CBOW -> M25CSA031_gensim_cbow.model")
    if best_sg:
        best_sg.save("M25CSA031_gensim_skipgram.model")
        print(f"  Saved baseline Skip-gram -> M25CSA031_gensim_skipgram.model")

    return gensim_models


def training_scratch_models(sentences):
    """
    Train a Word2vec model from scratch using numpy.    

    Training a CBOW and skip-gram with the configurationand comparing the quality of results 
    embeddings with gensim's embeddings.

    """
    print("\n" + "=" * 65)
    print(" SCRATCH MODEL TRAINING")
    print("=" * 65)

    scratch_models = {}

    # Train Skip-gram from scratch
    print("\n[1/2] Training Skip-gram from scratch...")
    sg_model = ScratchWord2Vec(
        sentences=sentences,
        embedding_dim=100,
        window_size=5,
        negative_samples=5,
        min_count=2,
        learning_rate=0.025,
        epochs=5
    )
    sg_model.train_skipgram()
    scratch_models["scratch_skipgram"] = sg_model

    # Train CBOW from scratch
    print("\n[2/2] Training CBOW from scratch...")
    cbow_model = ScratchWord2Vec(
        sentences=sentences,
        embedding_dim=100,
        window_size=5,
        negative_samples=5,
        min_count=2,
        learning_rate=0.025,
        epochs=5
    )
    cbow_model.train_cbow()
    scratch_models["scratch_cbow"] = cbow_model

    # Save the scratch models using pickle
    for name, model in scratch_models.items():
        save_path = f"M25CSA031_{name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  [*] Saved {name} → {save_path}")

    return scratch_models


def comparing_models(gensim_models, scratch_models):
    """
    Comparing as showing a top 5 neighbors for a test word from both gensim and 
    scratch models.

    """
    test_words = ["research", "student", "faculty"]

    print("\n" + "=" * 65)
    print("  COMPARISON: SCRATCH vs GENSIM")
    print("=" * 65)

    for word in test_words:
        print(f"\n  Query word: '{word}'")
        print(f"  {'Model':<25s} | {'Neighbors'}")
        print(f"  {'-'*25}-+-{'-'*50}")

        # Gensim CBOW baseline
        gensim_cbow = gensim_models.get("CBOW_d100_w5_n5")
        if gensim_cbow and word in gensim_cbow.wv:
            nn = gensim_cbow.wv.most_similar(word, topn=5)
            nn_str = ", ".join([f"{w}({s:.2f})" for w, s in nn])
            print(f"  {'Gensim CBOW':<25s} | {nn_str}")
        
        # Gensim Skip-gram baseline
        gensim_sg = gensim_models.get("Skip-gram_d100_w5_n5")
        if gensim_sg and word in gensim_sg.wv:
            nn = gensim_sg.wv.most_similar(word, topn=5)
            nn_str = ", ".join([f"{w}({s:.2f})" for w, s in nn])
            print(f"  {'Gensim Skip-gram':<25s} | {nn_str}")

        # Scratch CBOW
        scratch_cbow = scratch_models.get("scratch_cbow")
        if scratch_cbow:
            nn = scratch_cbow.most_similar(word, top_n=5)
            if nn:
                nn_str = ", ".join([f"{w}({s:.2f})" for w, s in nn])
                print(f"  {'Scratch CBOW':<25s} | {nn_str}")

        # Scratch Skip-gram
        scratch_sg = scratch_models.get("scratch_skipgram")
        if scratch_sg:
            nn = scratch_sg.most_similar(word, top_n=5)
            if nn:
                nn_str = ", ".join([f"{w}({s:.2f})" for w, s in nn])
                print(f"  {'Scratch Skip-gram':<25s} | {nn_str}")


# Main Execution
if __name__ == "__main__":
    CORPUS_FILE = "clened_corpus.txt"

    print("=" * 65)
    print("  WORD2VEC TRAINING")
    print("=" * 65)

    sentences = corpus_loader(CORPUS_FILE)

    if not sentences:
        print("No data! Run data_preparation.py first.")
    else:
        # Train all Gensim models
        gensim_models = training_gensim_models(sentences)

        # Train scratch models
        scratch_models = training_scratch_models(sentences)

        # Compare scratch vs Gensim side by side
        comparing_models(gensim_models, scratch_models)

        print("\n\nAll training complete!")
