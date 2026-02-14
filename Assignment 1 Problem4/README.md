# Sport vs Politics Text Classification

A project for binary classification on the sport vs politics datasets.

## Project Overview

This project implements and compares three machine learning techniques (Naive Bayes, Logistic Regression, and SVM) combined with three feature representation methods (Bag of Words, TF-IDF, and N-grams) for classifying text documents as either Sport or Politics.

**Best Performance:** Naive Bayes with N-grams achieves **95.37% accuracy** and **96.00% F1-score**.

## Dataset

- **Source:** 20 Newsgroups dataset
- **Total Articles:** 4,494
  - Sport: 1,933 articles (43%)
  - Politics: 2,561 articles (57%)
- **Split:** 80% training (3,595), 20% testing (899)
- **Categories:**
  - Sport: rec.sport.baseball, rec.sport.hockey
  - Politics: talk.politics.guns, talk.politics.mideast, talk.politics.misc

### Prerequisites

```bash
pip install numpy scikit-learn matplotlib
```

### Running the Classifier

```bash
python M25CSA031_prob4.py
```

This will:

1. Load and preprocess the dataset
2. Train all 9 model combinations
3. Evaluate performance on test set
4. Generate visualizations and report

## Methodology

### Text Preprocessing

1. Lowercase conversion
2. URL and email removal
3. Special character removal
4. Whitespace normalization

### Feature Representations

**1. Bag of Words**

- Word frequency vectors
- Max features: 5,000
- Min document frequency: 2

**2. TF-IDF**

- Term frequency inverse document frequency weighting
- Emphasizes discriminative terms
- Sublinear TF scaling

**3. N-grams**

- Unigrams and bigrams
- Captures contextual information
- Combined with TF-IDF weighting

### Machine Learning Algorithms

**1. Naive Bayes**

- Multinomial Naive Bayes
- Laplace smoothing
- Fastest training time

**2. Logistic Regression**

- L2 regularization
- LBFGS solver
- Max iterations: 1,000

**3. Support Vector Machine**

- Linear kernel
- Squared hinge loss
- Regularization: C=1.0

## Results

### Performance Comparison

| ML Technique    | Features    | Accuracy   | Precision  | Recall     | F1-Score   |
| --------------- | ----------- | ---------- | ---------- | ---------- | ---------- |
| **Naive Bayes** | **N-grams** | **95.37%** | **94.69%** | **97.35%** | **96.00%** |
| Naive Bayes     | TF-IDF      | 95.37%     | 94.82%     | 97.19%     | 95.99%     |
| Naive Bayes     | BoW         | 95.37%     | 95.38%     | 96.57%     | 95.97%     |
| SVM             | TF-IDF      | 94.48%     | 93.67%     | 96.88%     | 95.25%     |
| Log. Regression | TF-IDF      | 94.13%     | 91.85%     | 98.44%     | 95.01%     |
| SVM             | N-grams     | 93.88%     | 92.59%     | 97.66%     | 95.06%     |
| Log. Regression | N-grams     | 93.77%     | 92.31%     | 97.66%     | 94.91%     |
| SVM             | BoW         | 93.55%     | 91.67%     | 98.05%     | 94.75%     |
| Log. Regression | BoW         | 93.10%     | 90.74%     | 98.44%     | 94.43%     |

### Key Findings

1. **Best Model:** Naive Bayes with N-grams (95.37% accuracy)
2. **Fastest Training:** Naive Bayes (0.002-0.004 seconds)
3. **Most Consistent:** TF-IDF features work well across all algorithms
4. **All Models:** Achieve >93% accuracy, demonstrating task feasibility

## Visualizations

- Confusion matrices for top 3 models
- Performance comparison charts
- Detailed evaluation report (`evaluation_report.txt`)

## Limitations

### Dataset Limitations

- Binary classification only
- English language only
- Forum discussions
- Temporal bias

### Technical Limitations

- Limited to 5,000 features
- No semantic understanding
- Linear decision boundaries only
- No deep learning approaches

### Model Limitations

- Independence assumption
- Fixed vocabulary
- No transfer learning
- Requires retraining for new domains

## Future Improvements

- Implement deep learning models
- Extend to multi-class classification
- Add multilingual support
- Incorporate word embeddings
- Develop ensemble methods

## License

This project is for academic purposes only.
