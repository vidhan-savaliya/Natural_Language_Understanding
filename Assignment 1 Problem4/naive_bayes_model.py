"""
NAIVE BAYES CLASSIFIER MODULE


This part is implementing a naive bias for the text classification problem.

"""

import time
from sklearn.naive_bayes import MultinomialNB


def naive_bayestraining(X_train, y_train, alpha = 1.0):
    """
    Training of a Naive Bayes classifier.
    

    Args:
        X_train : Training feature matrix (sparse matrix from vectorizer)
        y_train : Training labels (0 for Sport, 1 for Politics)
        alpha : Laplace smoothing parameter 
        
    Returns:
        tuple: 
            - trainedmodel : Fitted MultinomialNB classifier
            - trainingtime : Time taken to train in seconds
            
   
    """
    
    print("Training naive bias classifier.....")
    print("*" * 60)
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Features: {X_train.shape[1]:,}")
    print(f"Smoothing parameter : {alpha}")
    print()
    
    # Recording a starting time for performance measure
    start_time = time.time()
    
    # Initializing Naive Bayes classifier
    #for handles a xero probability alpha = 1.0 is Laplace smoothing
    # ensuring that unseen words don't cause probability to be zero

    model = MultinomialNB(alpha=alpha)
    
    # Training the model
    # The model learns:
    # 1.Prior probabilities :- P(Sport), P(Politics)
    # 2.Likelihood probabilities :- P(word|Sport), P(word|Politics)
    model.fit(X_train, y_train)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    print(f" Training completed in {training_time:.4f} seconds")
    print(f" Model is ready for prediction")
    print()
    
    return model, training_time


def modelinformation(model, feature_names=None, top_n=10):
    """
    extracting information of the model from trained naive bayes.
    
    In this function highlights which word is indicative of both classes,
    helping for inderstanding what model is learned.
    
    Args:
        model : Trained MultinomialNB model
        feature_names : List of feature names
        top_n : Number of top features to show per class
        
    Returns:
        dict : Dictionary with model information
    """
    info = {
        'n_features': model.feature_count_.shape[1],
        'class_prior': model.class_log_prior_,
        'classes': model.classes_
    }
    
    if feature_names is not None:

        # Getting word probabilities of each word from each class

        feature_log_prob = model.feature_log_prob_
        
        # Top words for Sport
        top_indices_sport = feature_log_prob[0].argsort()[-top_n:][::-1]
        top_words_sport = [feature_names[i] for i in top_indices_sport]
        
        # Top words for Politics
        top_indices_politics = feature_log_prob[1].argsort()[-top_n:][::-1]
        top_words_politics = [feature_names[i] for i in top_indices_politics]
        
        info['sportindicators'] = top_words_sport
        info['politicsindicators'] = top_words_politics
    
    return info


if __name__ == "__main__":
    print("Testing Naive Bayes classifier module.....\n")
    
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    
    # Sample training data
    train_docs = [
        "india won cricket world cup final match today",
        "virat kohli scored century in ipl tournament",
        "rohit sharma leads mumbai indians to victory",
        "prime minister announced new digital india policy",
        "parliament passed education reform bill today",
        "lok sabha discussed agriculture laws debate"
    ]
    
    train_labels = np.array([0, 0, 0, 1, 1, 1])  # 0=Sport, 1=Politics
    
    print("=" * 80)
    print("SAMPLE TRAINING DATA (Indian Context)")
    print("=" * 80)
    print("\nSport documents:")
    for i, doc in enumerate(train_docs[:3]):
        print(f"  {i+1}. {doc}")
    print("\nPolitics documents:")
    for i, doc in enumerate(train_docs[3:]):
        print(f"  {i+1}. {doc}")
    print()
    
    # Vectorize the documents
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_docs)
    
    # Training the model
    model, train_time = naive_bayestraining(X_train, train_labels)
    
    # Get model information
    feature_names = vectorizer.get_feature_names_out()
    info = modelinformation(model, feature_names, top_n=5)
    
    
    print("MODEL INFORMATION")
    print("*" * 60)
    print(f"Number of features: {info['n_features']}")
    print(f"Number of classes: {info['n_classes']}")
    
    print(f"\nTop Sport indicators: {', '.join(info['sportindicators'])}")
    print(f"Top Politics indicators: {', '.join(info['politicsindicators'])}")
    print()
    
    # Test prediction
    test_docs = [
        "indian cricket team wins test series",
        "government announces new policy reforms"
    ]
    X_test = vectorizer.transform(test_docs)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    
    print("TEST PREDICTIONS")
    print("*" * 60)
    for i, doc in enumerate(test_docs):

        pred_class = "Sport" if predictions[i] == 0 else "Politics"
        confidence = probabilities[i][predictions[i]] * 100

        print(f"\nDocument: '{doc}'")

        print(f"Prediction: {pred_class} (confidence: {confidence:.1f}%)")
    
    
    print("Naive Bayes module test completed.")
    print("*" * 60)
