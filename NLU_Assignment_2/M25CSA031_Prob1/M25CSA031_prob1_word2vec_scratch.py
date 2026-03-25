"""
Problem 1: Word2Vec Implementation from Scratch

Complete Word2vec model using a numpy.
not used any predefined libraries.

Implemented CBOW and skip-gram model.

Both uses Negative Sampling to avoid softmax over entire vocabulary.


The key equations:
For a positive pair:   loss += -log(sigmoid(dot(u_center, v_context)))
For a negative pair:   loss += -log(sigmoid(-dot(u_center, v_negative)))
Gradients are computed analytically and applied via vanilla SGD.

"""

import numpy as np
from collections import Counter
import random
import time


class ScratchWord2Vec:
    """
    Train word embeddings using Word2Vec algorithm.

    Working:
    Vocabulary building 
    W_input and W_output matrix creation
    for each training window in the text:
        center word and context words
        negative samples
        computes a gradients
        updates W_input and W_output
    after training, W_input contains the final word embeddings

    """

    def __init__(self, sentences, embedding_dim=100, window_size=5, 
                 negative_samples=5, min_count=2, learning_rate=0.025, epochs=5):
        """

        Parameters:

        sentences : list of list of str
            Tokenized corpus. Each sentence is a list of words.

        embedding_dim : int
            How many dimensions each word vector should have

        window_size : int
            How many words to the left and right to consider as context

        negative_samples : int
            How many random noise words to sample per positive pair

        min_count : int
            Ignore words that appear fewer than this many times

        learning_rate : float
            Step size for SGD updates

        epochs : int
            Number of full passes over the training data
        """
        self.dim = embedding_dim
        self.window = window_size
        self.neg_k = negative_samples
        self.lr = learning_rate
        self.epochs = epochs
        self.min_count = min_count

        # Building the vocabulary from the raw sentences
        # We count every word, then keep only those appearing >= min_count times
        self.build_vocab(sentences)

        # Converting sentences from strings to integer IDs for fast lookup
        self.prepare_training_data(sentences)

        # Building the negative sampling probability table
        # As the suggenstion of mikolov using of unigram raised to 0.75 power
        # Thats why less frequent words a highr chance for sampled
        self.build_sampling_table()

        # Initializing the two weight matrices

        # W_input: the actual word embeddings we care about
        # W_output: the context weight matrix used only during training

        # Random init for W_input, zero init for W_output
        self.W_input = (np.random.rand(self.vocab_size, self.dim) - 0.5) / self.dim
        self.W_output = np.zeros((self.vocab_size, self.dim))

    def build_vocab(self, sentences):
        """
        Counting all words and assign integer IDs only to the frequents.
        this is because rare words dont have enough context to learn meaningful embeddings

        """
        # count everything in first pass
        word_counts = Counter()
        for sentence in sentences:
            for word in sentence:
                word_counts[word] += 1

        # keeping only words with count >= min_count
        self.word2id = {}
        self.id2word = {}
        self.word_freq = []
        
        idx = 0
        for word, count in word_counts.items():
            if count >= self.min_count:
                self.word2id[word] = idx
                self.id2word[idx] = word
                self.word_freq.append(count)
                idx += 1

        self.vocab_size = len(self.word2id)
        print(f" Vocabulary built: {self.vocab_size} words "
              f"(dropped {len(word_counts) - self.vocab_size} rare words)")

    def prepare_training_data(self, sentences):
        """
        training data preparation- converting each sentence into list of integer IDs.
        Skip any word not in vocabulary.

        
        """
        self.training_data = []
        for sentence in sentences:
            id_sentence = []
            for word in sentence:
                if word in self.word2id:
                    id_sentence.append(self.word2id[word])
            # Only keep sentences with at least 2 tokens 
            if len(id_sentence) >= 2:
                self.training_data.append(id_sentence)

        total_tokens = sum(len(s) for s in self.training_data)
        print(f" Training data: {len(self.training_data)} sentences, "
              f"{total_tokens} tokens total")

    def build_sampling_table(self):
        """

        Building a probability distribution for negative sampling.
        
        Instead of uniform random sampling, we raise each word's frequency to the power of 0.75.
        This makes common words slightly less dominant and gives rare words a fair chance of being picked as negatives.
        
        P(w_i) = freq(w_i)^0.75 / sum_j(freq(w_j)^0.75)
        """
        freqs = np.array(self.word_freq, dtype=np.float64)
        # Raise to 0.75 power from paper
        powered = np.power(freqs, 0.75)

        # Normalize to make it a valid probability distribution
        self.neg_probs = powered / powered.sum()

    def sample_negatives(self, exclude_ids):
        """
        Sampling a K random word IDs as negative examples.
        None of them are actual context words.


        Parameters:
        
        exclude_ids : set of int
            Word IDs to avoid
        """
        negatives = []
        while len(negatives) < self.neg_k:
            # np.random.choice with our custom probability table
            sampled = np.random.choice(self.vocab_size, p=self.neg_probs)
            if sampled not in exclude_ids:
                negatives.append(sampled)
        return negatives

    def sigmoid(self, x):
        """
        Standard sigmoid function: sigma(x) = 1 / (1 + exp(-x))
        We clipping x to avoid numerical overflow in exp()
        """
        x = np.clip(x, -8, 8)
        return 1.0 / (1.0 + np.exp(-x))

    def train_skipgram(self):
        """
        Skip-gram training
        
        each position in the text, the CENTER word tries 
        to predict each CONTEXT word independently.
        
        For each positive pair (center, context), we also sample K negative 
        words and do a binary classification update.
        """
        print(f"\n  [Skip-gram] Starting training: dim={self.dim}, window={self.window}, "
              f"neg={self.neg_k}, epochs={self.epochs}")
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            pairs_trained = 0
            t0 = time.time()

            # Shuffled a sentences each epoch for better SGD convergence
            random.shuffle(self.training_data)

            for sentence in self.training_data:
                sent_len = len(sentence)

                for center_pos in range(sent_len):
                    center_id = sentence[center_pos]
                    
                    # Defining the context window boundaries
                    # we randomly shrink the window slightly for variety
                    actual_window = random.randint(1, self.window)
                    start = max(0, center_pos - actual_window)
                    end = min(sent_len, center_pos + actual_window + 1)

                    for ctx_pos in range(start, end):
                        if ctx_pos == center_pos:
                            continue  # skip the center word itself

                        context_id = sentence[ctx_pos]
                        
                        # Get negative samples
                        negatives = self.sample_negatives({center_id, context_id})

                        # Forward pass & backward pass

                        # Look up the center word's embedding
                        h = self.W_input[center_id].copy()
                        
                        grad_input = np.zeros(self.dim)

                        # POSITIVE example: context word should have label = 1
                        dot_pos = np.dot(h, self.W_output[context_id])
                        sig_pos = self.sigmoid(dot_pos)
                        error_pos = sig_pos - 1.0  # gradient for positive
                        
                        # Accumulate gradient for center embedding
                        grad_input += error_pos * self.W_output[context_id]
                        # Update context word's output vector
                        self.W_output[context_id] -= self.lr * error_pos * h
                        
                        # Track loss: -log(sigmoid(dot_pos))
                        epoch_loss += -np.log(max(sig_pos, 1e-10))

                        # NEGATIVE examples: random words should have label = 0
                        for neg_id in negatives:
                            dot_neg = np.dot(h, self.W_output[neg_id])
                            sig_neg = self.sigmoid(dot_neg)
                            error_neg = sig_neg  # gradient for negative
                            
                            grad_input += error_neg * self.W_output[neg_id]
                            self.W_output[neg_id] -= self.lr * error_neg * h
                            
                            epoch_loss += -np.log(max(1.0 - sig_neg, 1e-10))

                        # Finally update the center word's input embedding
                        self.W_input[center_id] -= self.lr * grad_input
                        pairs_trained += 1

            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(pairs_trained, 1)
            print(f"    Epoch {epoch+1}/{self.epochs} — Loss: {avg_loss:.4f} | "
                  f"Pairs: {pairs_trained:,} | Time: {elapsed:.1f}s")

        print("  [Skip-gram] Training complete!")

    def train_cbow(self):
        """

        CBOW training.
        
        In this opposite of the skip-gram model: we took all context of words in window and average their embeddings,
        then use that average to predict the center word.
        
    
        The gradient flows back through the average to update each context word.
        """
        print(f"\n  [CBOW] Starting training: dim={self.dim}, window={self.window}, "
              f"neg={self.neg_k}, epochs={self.epochs}")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            pairs_trained = 0
            t0 = time.time()

            random.shuffle(self.training_data)

            for sentence in self.training_data:
                sent_len = len(sentence)

                for center_pos in range(sent_len):
                    center_id = sentence[center_pos]

                    actual_window = random.randint(1, self.window)
                    start = max(0, center_pos - actual_window)
                    end = min(sent_len, center_pos + actual_window + 1)

                    # Collect all context word IDs
                    context_ids = []
                    for ctx_pos in range(start, end):
                        if ctx_pos != center_pos:
                            context_ids.append(sentence[ctx_pos])

                    if len(context_ids) == 0:
                        continue

                    # CBOW hidden layer = AVERAGE of all context word embeddings
                    h = np.mean(self.W_input[context_ids], axis=0)

                    negatives = self.sample_negatives(set(context_ids + [center_id]))

                    grad_h = np.zeros(self.dim)

                    # POSITIVE: center word
                    dot_pos = np.dot(h, self.W_output[center_id])
                    sig_pos = self.sigmoid(dot_pos)
                    error_pos = sig_pos - 1.0
                    grad_h += error_pos * self.W_output[center_id]
                    self.W_output[center_id] -= self.lr * error_pos * h
                    epoch_loss += -np.log(max(sig_pos, 1e-10))

                    # NEGATIVES
                    for neg_id in negatives:
                        dot_neg = np.dot(h, self.W_output[neg_id])
                        sig_neg = self.sigmoid(dot_neg)
                        error_neg = sig_neg
                        grad_h += error_neg * self.W_output[neg_id]
                        self.W_output[neg_id] -= self.lr * error_neg * h
                        epoch_loss += -np.log(max(1.0 - sig_neg, 1e-10))

                    # Distribute the gradient equally to each context word because h was the average, each context contributes 1/N
                    grad_per_ctx = grad_h / len(context_ids)
                    for cid in context_ids:
                        self.W_input[cid] -= self.lr * grad_per_ctx

                    pairs_trained += 1

            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(pairs_trained, 1)
            print(f"    Epoch {epoch+1}/{self.epochs} — Loss: {avg_loss:.4f} | "
                  f"Windows: {pairs_trained:,} | Time: {elapsed:.1f}s")

        print("  [CBOW] Training complete!")

    def get_embedding(self, word):
        """Return the learned vector for a word, or None if not in vocab."""
        if word in self.word2id:
            return self.W_input[self.word2id[word]]
        return None

    def most_similar(self, word, top_n=5):
        """
        Find the top_n most similar words using cosine similarity.
        
        Cosine similarity = dot(A, B) / (||A|| * ||B||)

        It measures the angle between two vectors, closer to 1.0 means the words appear in very similar contexts and are thus semantically related.
        """
        if word not in self.word2id:
            return []

        word_vec = self.W_input[self.word2id[word]]
        word_norm = np.linalg.norm(word_vec)
        
        if word_norm == 0:
            return []

        # Computing aosine similarities against the entire vocabular

        # its matrics vecot dot product
        all_norms = np.linalg.norm(self.W_input, axis=1)

        # Avoids division by zero
        all_norms = np.where(all_norms == 0, 1e-10, all_norms)
        
        similarities = np.dot(self.W_input, word_vec) / (all_norms * word_norm)

        # Getting indices sorted by similarity
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            if idx == self.word2id[word]:
                continue  # skip the query word itself
            results.append((self.id2word[idx], float(similarities[idx])))
            if len(results) >= top_n:
                break

        return results

    def analogy(self, word_a, word_b, word_c, top_n=1):
        """
        Solving analogies using vector arithmetic.
        
        A is to B as C is to ?
        
        
        The vector arithmetic is result = B - A + C
        and from that finding the closest word to this result vector.
        
        example: ug - btech + pg ≈ mtech?

        """
        for w in [word_a, word_b, word_c]:
            if w not in self.word2id:
                return []

        vec = (self.W_input[self.word2id[word_b]] 
               - self.W_input[self.word2id[word_a]] 
               + self.W_input[self.word2id[word_c]])

        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            return []

        all_norms = np.linalg.norm(self.W_input, axis=1)
        all_norms = np.where(all_norms == 0, 1e-10, all_norms)
        
        similarities = np.dot(self.W_input, vec) / (all_norms * vec_norm)

        # Excluding the three input words from results
        exclude = {self.word2id[word_a], self.word2id[word_b], self.word2id[word_c]}
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            if idx in exclude:
                continue
            results.append((self.id2word[idx], float(similarities[idx])))
            if len(results) >= top_n:
                break

        return results
