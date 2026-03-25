"""
Problem 1 : Semantic Analysis

In this evaluating the trained Word2Vec models using:
    finding top 5 nearest neighbours
    performing word analogy experiments

Test on both model trained using gensim and from scratch


"""

import pickle
from gensim.models import Word2Vec as GensimWord2Vec


def loads_models():
    """Loads both Gensim and scratch models from disk."""
    models = {}
    
    # Load Gensim models
    try:
        models["Gensim CBOW"] = ("gensim", GensimWord2Vec.load("M25CSA031_gensim_cbow.model"))
        print("  Loaded: Gensim CBOW")
    except FileNotFoundError:
        print("  WARNING: Gensim CBOW model not found")

    try:
        models["Gensim Skip-gram"] = ("gensim", GensimWord2Vec.load("M25CSA031_gensim_skipgram.model"))
        print("  Loaded: Gensim Skip-gram")
    except FileNotFoundError:
        print("  WARNING: Gensim Skip-gram model not found")

    # Load scratch models
    try:
        with open("M25CSA031_scratch_cbow.pkl", "rb") as f:
            models["Scratch CBOW"] = ("scratch", pickle.load(f))
        print("  Loaded: Scratch CBOW")
    except FileNotFoundError:
        print("  WARNING: Scratch CBOW model not found")

    try:
        with open("M25CSA031_scratch_skipgram.pkl", "rb") as f:
            models["Scratch Skip-gram"] = ("scratch", pickle.load(f))
        print("  Loaded: Scratch Skip-gram")
    except FileNotFoundError:
        print("  WARNING: Scratch Skip-gram model not found")

    return models


def getting_neighbors(model_type, model, word, top_n=5):
    """
    get a top n most similar words with the use of cosine simiarity.
    It can working with both the models.

    """
    if model_type == "gensim":
        try:
            return model.wv.most_similar(word, topn=top_n)
        except KeyError:
            return None
    else:
        # scratch model
        result = model.most_similar(word, top_n=top_n)
        return result if result else None


def getting_analogy(model_type, model, word_a, word_b, word_c, top_n=1):
    """
    get a analogy for the given words.
    It can working with both the models.
    """
    if model_type == "gensim":
        try:
            result = model.wv.most_similar(
                positive=[word_b, word_c], 
                negative=[word_a], 
                topn=top_n
            )
            return result
        except KeyError:
            return None
    else:
        result = model.analogy(word_a, word_b, word_c, top_n=top_n)
        return result if result else None


def nearest_neighbors(models):
    """
    top 5 nearest neighbors for required words.
    
    """
    target_words = ['research', 'student', 'phd', 'exam']

    print("\n" + "=" * 65)
    print(" TOP 5 NEAREST NEIGHBORS")
    print("=" * 65)

    for word in target_words:
        print(f"\n  >>> Target word: '{word}'")
        
        for model_name, (model_type, model) in models.items():
            neighbors = getting_neighbors(model_type, model, word)
            
            if neighbors:
                # Format: word1 (0.85), word2 (0.82), ...
                nn_str = ", ".join([f"{w} ({s:.3f})" for w, s in neighbors])
                print(f"    [{model_name}]: {nn_str}")
            else:
                print(f"    [{model_name}]: '{word}' not in vocabulary")


def run_analogy_tests(models):
    """
    Performing analogy experiments.
    
    Analogy working as:
    If A is to B as C is to ?, then ? = B - A + C
    
    For example: "king" is to "queen" as "man" is to ? → "woman"
    Because: queen_vec - king_vec + man_vec ≈ woman_vec
    
    In this analogies are tailored to iit jodhpur academic context.
    """
    analogies = [
        # (A, B, C) → A is to B as C is to ?
        ("ug", "btech", "pg", "Expected: mtech or phd"),
        ("student", "learning", "faculty", "Expected: teaching or research"),
        ("btech", "engineering", "msc", "Expected: science"),
    ]

    print("\n\n" + "=" * 65)
    print(" WORD ANALOGY EXPERIMENTS")
    print("=" * 65)

    for word_a, word_b, word_c, expected in analogies:
        print(f"\n  Analogy: {word_a} : {word_b} :: {word_c} : ?")
        print(f"  ({expected})")
        
        for model_name, (model_type, model) in models.items():
            result = getting_analogy(model_type, model, word_a, word_b, word_c, top_n=3)
            
            if result:
                res_str = ", ".join([f"{w} ({s:.3f})" for w, s in result])
                print(f"    [{model_name}]: {res_str}")
            else:
                print(f"    [{model_name}]: One or more words not in vocabulary")



# Main Execution
if __name__ == "__main__":
    print("=" * 65)
    print(" SEMANTIC ANALYSIS ")
    print("=" * 65)
    print("\nLoading trained models...")

    models = loads_models()

    if not models:
        print("No models found! Please run M25CSA031_prob1_training.py first.")
    else:
        nearest_neighbors(models)
        run_analogy_tests(models)
        print("\nSemantic analysis complete!")
