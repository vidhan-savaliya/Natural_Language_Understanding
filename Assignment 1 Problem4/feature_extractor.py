"""
FEATURE EXTRACTION MODULE


THis section is implementing a tree different type of feature etraction tecchniques:

1.Bag of Words
2.TF-IDF 
3.N-grams 

"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def bagofword_features(X_train, X_test, max_features=5000):
    """
    Extracting Bag of Words features from  documents.
    
    
    EXAMPLE:

    Document : "The team won the game"
    Vocabulary : ["team", "won", "game", "player", "score"]

    BoW Vector : [1, 1, 1, 0, 0]
               Means-> team=1, won=1, game=1, player=0, score=0
    

    
    Args:
        X_train : Training documents    
        X_test : Test documents 
        max_features : Maximum number of features
        
    Returns:
        tuple: (X_train_bow, X_test_bow, vectorizer)
            -> X_train_bow :- training feature matrix
            -> X_test_bow :- testing feature matrix
            -> vectorizer :- fitted countvectorizer object
    """
    print(f"Extracting Bag of Words features.....")
    
    # CountVectorizer is converting a text into word count matrix
    # max_features :- limits a vocabulary to most frequent words
    # min_df :- ignoring words appearing in less than 2 documents
    # stop_words='english' :- removes a common English words 

    vectorizer = CountVectorizer(max_features=max_features, min_df=2, stop_words='english')
    
    # Fit on training data and transform both train and test
    # fit_transform :- learns vocabulary from training data and transforms it
    # transform :- applies learned vocabulary to test data 

    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    
    print(f"BoW feature matrix shape: {X_train_bow.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Sparsity: {(1.0 - X_train_bow.nnz / (X_train_bow.shape[0] * X_train_bow.shape[1])) * 100:.2f}%")
    print()
    
    return X_train_bow, X_test_bow, vectorizer


def tfidf_features(X_train, X_test, max_features=5000):
    """
    Extracting TF-IDF features.
    

    Reducing weights of common words and increasing a weights of different words.

    
    EXAMPLE:

    IF Word->  "touchdown" is appears 5 times in a sports article but only in 10/1000 documents:
    - TF = 5 
    - IDF = log(1000/10) = 2.0
    - TF-IDF = 0.7 × 2.0 = 1.4 
    
   
    
    
    Args:
        X_train :- Training documents
        X_test :- Test documents
        max_features :- Maximum number of features
        
    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, vectorizer)
    """
    print(f"Extracting TF-IDF features.....")
    
    # TfidfVectorizer converting a text to TFIDF weighted matrix
    # sublinear_tf=True: uses logarithmic term frequency scaling (1 + log(TF))
    # This prevents very frequent words from dominating
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=2, stop_words='english',sublinear_tf=True)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    print(f"Sparsity: {(1.0 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])) * 100:.2f}%")
    print()
    
    return X_train_tfidf, X_test_tfidf, vectorizer


def ngram_features(X_train, X_test, ngram_range=(1, 2), max_features=5000):
    """
    Extracting N-gram features from text documents.
    
    
    
    SPORT BIGRAM EXAMPLES:
    - "cricket match", "world cup", "test series", "ipl match"
    - "final score", "winning goal", "championship game", "virat kohli"
    
    POLITICS BIGRAM EXAMPLES:
    - "prime minister", "lok sabha", "parliament session", "education policy"
    - "agriculture bill", "reservation policy", "supreme court", "voting rights"


    
    Args:
        X_train :- Training documents
        X_test :- Test documents
        ngram_range :- min_n--> max_n for n-gram range
        max_features :- Maximum number of features
        
    Returns:

        tuple: (X_train_ngram, X_test_ngram, vectorizer)
    """
    print(f"Extracting N-gram features.....")
    
    # Vectorizer with ngram_range extracts n-grams and weights them with TF-IDF

    # ngram_range => (1,2) means both unigrams and bigrams

    vectorizer = TfidfVectorizer(max_features=max_features, min_df=2, stop_words='english',ngram_range=ngram_range,sublinear_tf=True)
    
    X_train_ngram = vectorizer.fit_transform(X_train)
    X_test_ngram = vectorizer.transform(X_test)
    
    print(f"N-gram feature matrix shape: {X_train_ngram.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Sparsity: {(1.0 - X_train_ngram.nnz / (X_train_ngram.shape[0] * X_train_ngram.shape[1])) * 100:.2f}%")
    
    # Show some example n-grams
    feature_names = vectorizer.get_feature_names_out()
    bigrams = [f for f in feature_names if ' ' in f][:10]
    if bigrams:
        print(f"Sample bigrams: {', '.join(bigrams[:5])}")
    print()
    
    return X_train_ngram, X_test_ngram, vectorizer


if __name__ == "__main__":
    # Test the feature extraction module
    print("Testing feature extraction module...\n")
    
    # Sample documents
    train_docs = [
        "india won the cricket world cup final match today",
        "virat kohli scored century in ipl match some day",
        "prime minister announced new ai policy reforms",
        "parliament passed the Tech bill after debate"
    ]
    
    test_docs = [
        "indian cricket team wins test series against australia",
        "lok sabha discusses reservation policy in heated session"
    ]
    
    
    print("TESTING BAG OF WORDS")
    print("*" * 30)
    X_train_bow, X_test_bow, bow_vec = bagofword_features(train_docs, test_docs, max_features=50)
    
    
    print("TESTING TF-IDF")
    print("*" * 30)
    X_train_tfidf, X_test_tfidf, tfidf_vec = tfidf_features(train_docs, test_docs, max_features=50)
    
    
    print("TESTING N-GRAMS")
    print("*" * 30)
    
    X_train_ngram, X_test_ngram, ngram_vec = ngram_features(train_docs, test_docs, max_features=50)

