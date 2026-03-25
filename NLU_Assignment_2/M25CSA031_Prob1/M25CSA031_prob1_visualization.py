"""
Problem 1 : Visualization

In this we are visualizing the word embeddinngs which is projectected to 2D from 100D.

using PCA and t-SNE for dimensionality reduction.

Results:
    Gensim CBOW
    Gensim Skip-gram
    Scratch CBOW
    Scratch Skip-gram

In each image plot there is color coding of words based on their category.

"""

import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec as GensimWord2Vec


# Defining a group of words which relevant to IIT Jodhpur for clustering visualization.
# 3 categories of words
WORD_GROUPS = {
    "Academic Programs": {
        "words": ["btech", "mtech", "phd", "degree", "undergraduate", 
                   "postgraduate", "diploma", "program", "semester", "course"],
        "color": "#e74c3c"  # red
    },
    "People & Roles": {
        "words": ["student", "faculty", "professor", "director", "researcher",
                   "alumni", "scholar", "dean", "head", "coordinator"],
        "color": "#3498db"  # blue
    },
    "Departments & Fields": {
        "words": ["computer", "mechanical", "electrical", "civil", "chemical",
                   "physics", "mathematics", "biology", "engineering", "science"],
        "color": "#2ecc71"  # green
    }
}


def get_vectors_gensim(model, words):
    """Extracting a embedding vectors from a Gensim model for the given words."""
    valid_words = []
    vectors = []
    for w in words:
        if w in model.wv:
            valid_words.append(w)
            vectors.append(model.wv[w])
    return valid_words, np.array(vectors) if vectors else np.array([])


def get_vectors_scratch(model, words):
    """Extracting a embedding vectors from scratch implemented model for the given words."""
    valid_words = []
    vectors = []
    for w in words:
        vec = model.get_embedding(w)
        if vec is not None:
            valid_words.append(w)
            vectors.append(vec)
    return valid_words, np.array(vectors) if vectors else np.array([])


def make_plot(words, vectors, colors, title, filename):
    """
    It Creates a single 2D plot of words embeddings.

    Ploting with its label name colour by category.
    Thats why we are able to see different colors for different categories for clear plotting.
    
    """
    if len(vectors) < 3:
        print(f"  Skipping {filename} — too few valid words ({len(vectors)})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    for ax_idx, (method_name, method) in enumerate([
        ("PCA", PCA(n_components=2)),
        ("t-SNE", TSNE(n_components=2, perplexity=min(8, max(2, len(vectors)-1)), 
                       random_state=42, max_iter=1000))
    ]):
        coords = method.fit_transform(vectors)
        ax = axes[ax_idx]
        
        # Plot each point with its category color
        for i in range(len(words)):
            ax.scatter(coords[i, 0], coords[i, 1], 
                      color=colors[i], s=120, edgecolors='black', 
                      linewidths=0.5, alpha=0.8, zorder=3)
            # Adding the word label slightly offset so it doesn't overlap the dot
            ax.annotate(words[i], (coords[i, 0], coords[i, 1]),
                       fontsize=9, fontweight='bold',
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_title(f"{method_name} Projection", fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
    
    # Creates a legend showing what each color means
    legend_elements = []
    for group_name, group_info in WORD_GROUPS.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=group_info["color"], markersize=10,
                   label=group_name)
        )
    
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=3, fontsize=11, frameon=True)
    
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def run_visualization():
    """Generates all visualization plots for all models."""
    
    # Collect all words from our predefined groups
    all_target_words = []
    word_to_color = {}
    for group_name, group_info in WORD_GROUPS.items():
        for w in group_info["words"]:
            all_target_words.append(w)
            word_to_color[w] = group_info["color"]

    # Create output directory
    os.makedirs("results", exist_ok=True)

    # Gensim models
    print("\n Gensim Model Visualizations")
    for arch, filename in [("cbow", "M25CSA031_gensim_cbow.model"), 
                            ("skipgram", "M25CSA031_gensim_skipgram.model")]:
        try:
            model = GensimWord2Vec.load(filename)
            words, vecs = get_vectors_gensim(model, all_target_words)
            colors = [word_to_color[w] for w in words]
            
            title = f"Gensim {arch.upper()} Word Embeddings"
            out_file = f"results/M25CSA031_gensim_{arch}_visualization.png"
            make_plot(words, vecs, colors, title, out_file)
            
        except FileNotFoundError:
            print(f"  Skipping Gensim {arch} — model file not found")

    # Scratch models
    print("\n Scratch Model Visualizations")
    for arch, filename in [("cbow", "M25CSA031_scratch_cbow.pkl"), 
                            ("skipgram", "M25CSA031_scratch_skipgram.pkl")]:
        try:
            with open(filename, "rb") as f:
                model = pickle.load(f)
            words, vecs = get_vectors_scratch(model, all_target_words)
            colors = [word_to_color[w] for w in words]
            
            title = f"Scratch {arch.upper()} Word Embeddings"
            out_file = f"results/M25CSA031_scratch_{arch}_visualization.png"
            make_plot(words, vecs, colors, title, out_file)
            
        except FileNotFoundError:
            print(f"  Skipping Scratch {arch} — model file not found")


# Main
if __name__ == "__main__":
    print("=" * 65)
    print("  WORD EMBEDDING VISUALIZATION")
    print("=" * 65)
    
    run_visualization()
    print("\nVisualization complete!")
