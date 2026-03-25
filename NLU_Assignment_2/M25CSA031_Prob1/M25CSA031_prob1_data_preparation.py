"""
Problem 1 : Data Preparation & Statistics

This file is reading a lines from raw corpus file and cleaning it. Preprocessing and procucing a cleaned 
corpus for training. it is also generating word cloud image.


Preprocessing steps :
    Lowercasing - converting all the text to lowercase.
    Punctuation/artifact removal - regex strips non-alphabetic characters
    Tokenization - NLTK's word_tokenize splits text into individual words
    Stopword removal - common English words like "the", "is", "at" are removed
    Deduplication - exact duplicate sentences are removed to avoid model bias

"""

# importing required libraries
import re
import os
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Using non-interactive backend so it can work without a display
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Using NLTK because for tokenization and stopwords because, it is a wellknown standard lib which handles edge cases.

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Downloading required NLTK data files
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


def raw_corpus_loading(filepath):
    """
    Loading a raw corpus text file and returning a list of lines.
    that each line represents a paragraph ot doc from scraper.

    """
    if not os.path.exists(filepath):
        print(f"ERROR: Cannot find {filepath}!")
        print("Please run the scraper first or provide a raw_corpus.txt file.")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"Loaded {len(lines)} raw lines from {filepath}")
    return lines


def text_cleaning(lines):
    """
    Applying the full pipeline of preprocessing to raw corpus lines.
    
    Lowercasing, Removing punctuation, Stopword removal, Deduplication.

    """
    # Loading the standard english stopwords
    english_stops = set(stopwords.words('english'))
    # Adding website link specific noisy words
    english_stops.update(['http', 'https', 'www', 'com', 'org', 'iit', 'iitj', 'ac'])

    cleaned_sentences = []
    all_tokens = []        # for tracking every tokens 
    seen_sentences = set() # for deduplication

    for line in lines:
        line = line.strip()
        
        # skipping sort lines for the noise removal
        if len(line) < 15:
            continue

        # Converting to lowercase
        line = line.lower()

        # Removing everything that isn't a letter or space
        # for urls and special characters removal
        line = re.sub(r'[^a-z\s]', ' ', line)

        # Tokenize using NLTK 
        try:
            tokens = word_tokenize(line)
        except Exception:
            # Fallback to basic split if NLTK has issues
            tokens = line.split()

        # Removing stopwords and very short tokens 
        filtered = []
        for token in tokens:
            if token not in english_stops and len(token) > 2:
                filtered.append(token)
                all_tokens.append(token)  # track for statistics

        # Keeping sentences with at least 3 meaningful words
        if len(filtered) >= 3:
            sentence_str = " ".join(filtered)
            
            # skipping if we've seen this exact sentence before
            if sentence_str not in seen_sentences:
                seen_sentences.add(sentence_str)
                cleaned_sentences.append(sentence_str)

    return cleaned_sentences, all_tokens


def saving_corpus(sentences, filepath):
    """Writing cleaned sentences to a text file. """
    with open(filepath, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")
    print(f"Saved {len(sentences)} cleaned sentences to {filepath}")


def print_statistics(sentences, all_tokens):
    """
    Printing the dataset statistics:
        Total number of documents
        Total number of tokens
        Vocabulary size

    """
    vocab = set(all_tokens)
    
    print("\n" + "=" * 50)
    print(" DATASET STATISTICS")
    print("=" * 50)
    print(f" Total documents: {len(sentences)}")
    print(f" Total tokens: {len(all_tokens)}")
    print(f" Vocabulary size: {len(vocab)}")
    print("=" * 50)
    
    # top 20 words printing
    freq = Counter(all_tokens)
    print("\n  Top 20 most frequent words:")
    for rank, (word, count) in enumerate(freq.most_common(20), 1):
        print(f"    {rank:2d}. {word:<20s} — {count}")


def generating_wordcloud(all_tokens, save_path):
    """
    it creates a word cloud image showing most frequent words as bigger words has higher frequency of occurence.
    
    """
    freq = Counter(all_tokens)
    
    wc = WordCloud(
        width=1000, 
        height=500, 
        background_color='white',
        max_words=150,
        colormap='viridis'  # a nice modern color scheme
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of IIT Jodhpur Corpus", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Word Cloud saved to {save_path}")


# ===== Main execution =====
if __name__ == "__main__":
    # Input: the raw text file (either from scraper or manually provided)
    RAW_FILE = "raw_corpus.txt"
    
    # Output: ONE single cleaned corpus file (used by all other scripts)
    CLEAN_FILE = "clened_corpus.txt"
    
    # Output: Word Cloud image
    WORDCLOUD_FILE = "M25CSA031_prob1_wordcloud.png"

    print("=" * 50)
    print(" CORPUS PREPROCESSING")
    print("=" * 50)

    # Step 1: Load raw data
    raw_lines = raw_corpus_loading(RAW_FILE)
    
    if raw_lines:
        # Step 2: Clean and preprocess
        cleaned, all_tokens = text_cleaning(raw_lines)
        
        # Step 3: Save the cleaned corpus
        saving_corpus(cleaned, CLEAN_FILE)
        
        # Step 4: Print statistics
        print_statistics(cleaned, all_tokens)
        
        # Step 5: Generate Word Cloud
        generating_wordcloud(all_tokens, WORDCLOUD_FILE)
        
        print("\nPreprocessing complete! Use the cleaned corpus for training.")
