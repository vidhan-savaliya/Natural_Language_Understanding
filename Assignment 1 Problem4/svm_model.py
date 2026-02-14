"""
SUPPORT VECTOR MACHINE CLASSIFIER MODULE


This part is implementing SVM for classification of sports and politics classes,
finding a hyperplane which maximize a margib between sports and politics classes.



"""

import time
import numpy as np
from sklearn.svm import LinearSVC


def svm(X_train, y_train, C=1.0, max_iterations=1000):
    """
    Train a Support Vector Machine classifier.
    
    
    Args:
        X_train :- Training feature matrix
        y_train :- Training labels 
        C :- Regularization parameter 
        max_iterations :- Maximum number of iterations
        
    Returns:
        tuple: (trained_model, training_time)
            -> trained_model :- Fitted LinearSVC classifier
            -> training_time :- Time taken to train in seconds
            

    """
    
    print("Training SVM classifier....")
    print("*" * 80)

    print(f"Training samples : {X_train.shape[0]:,}")
    print(f"Features : {X_train.shape[1]:,}")

    print(f"Regularization parameter : {C}")
    print(f"Maximum iterations : {max_iterations}")
    print()
    
    # Start time for performance 
    start_time = time.time()
    
    # Linear SVM classifier

    # C = 1.0 :- regularization parameter 
    # max_iterations = 1000 :- maximum iterations for convergence
    # random_state = 42 :- for reproducibility

    # dual = 'auto' :- automatically choose primal or dual formulation
    #   -> Primal is faster when n_samples > n_features
    #   -> Dual is faster when n_samples < n_features
    # loss='squared_hinge' :- squared hinge loss function (smoother than hinge)
    model = LinearSVC(
        C=C,
        max_iter=max_iterations,
        random_state=42,
        dual='auto',
        loss='squared_hinge'
    )
    
    # Train the model
    # The model learns:
    # 1. Weight vector w (defines the hyperplane)
    # 2. Bias term b (offset from origin)
    # 3. Support vectors (critical data points)
    model.fit(X_train, y_train)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Check if model converged
    if model.n_iter_ >= max_iterations:
        print("Warning: Model may not have converged. Please Increase max_iterations.")
    else:
        print(f"Model converged in {model.n_iter_} iterations")
    
    print(f"Training completed in {training_time:.4f} seconds")
    print(f"Model is ready for prediction")
    print()
    
    return model, training_time


def svm_info(model, feature_names=None, top_n=20):
    """
    Extracting information from trained SVM model.
    
    Coefficients of svm similar to Logistic Regression:

    -> Positive coefficients :- words that push toward Politics
    -> Negative coefficients :- words that push toward Sport
    -> Larger absolute value :- stronger influence on decision
    
    Args:
        model: Trained LinearSVC model
        feature_names: List of feature names (words)
        top_n: Number of top features to show per class
        
    Returns:
        dict: Dictionary with model information

    """
    info = {
        'n_features': model.coef_.shape[1],
        'intercept': model.intercept_[0],
        'n_iter': model.n_iter_
    }
    
    if feature_names is not None:
        # Weights for features
        coefficients = model.coef_[0]
        
        # Top Sport indicators
        sport_indices = np.argsort(coefficients)[:top_n]
        sport_features = [(feature_names[i], coefficients[i]) for i in sport_indices]
        
        # Top Politics indicators
        politics_indices = np.argsort(coefficients)[-top_n:][::-1]
        politics_features = [(feature_names[i], coefficients[i]) for i in politics_indices]
        
        info['sport_indicators'] = sport_features
        info['politics_indicators'] = politics_features
    
    return info


if __name__ == "__main__":
    # Testing SVM classifier module
    print("Testing SVM classifier module...\n")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    
    # Samples of sports and politics documents
    train_docs = [
        "india won cricket world cup final match today",
        "virat kohli scored century in ipl tournament",
        "rohit sharma leads mumbai indians to victory",
        "prime minister announced new digital india policy",
        "parliament passed education reform bill today",
        "lok sabha discussed agriculture laws debate"
    ]
    
    train_labels = np.array([0, 0, 0, 1, 1, 1]) 
    
    print("SAMPLE TRAINING DATA")
    print("*" * 80)

    print("\nSport documents:")

    for i, doc in enumerate(train_docs[:3]):
        print(f"  {i+1}. {doc}")
    print("\nPolitics documents:")

    for i, doc in enumerate(train_docs[3:]):
        print(f"  {i+1}. {doc}")
    print()
    
    # Vectorize the documents
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_docs)
    
    # Training the model
    model, train_time = svm(X_train, train_labels)
    
    # Get model information
    feature_names = vectorizer.get_feature_names_out()
    info = svm_info(model, feature_names, top_n=5)
    

    print("MODEL INFORMATION")
    print("*" * 40)
    print(f"Number of features: {info['n_features']}")
    print(f"Support vectors: {info['support_vectors']}")
    print(f"\nTop Sport indicators:")

    for word, coef in info['sport_indicators']:
        print(f"  {word}: {coef:.4f}")
    print(f"\nTop Politics indicators:")
    
    for word, coef in info['politics_indicators']:
        print(f"  {word}: {coef:.4f}")
    print()
    
    # Test prediction
    test_docs = [
        "indian cricket team wins test series",
        "government announces new policy reforms"
    ]
    X_test = vectorizer.transform(test_docs)
    predictions = model.predict(X_test)
    decision_scores = model.decision_function(X_test)
    

    print("TEST PREDICTIONS")
    print("*" * 60)
    
    for i, doc in enumerate(test_docs):
        pred_class = "Sport" if predictions[i] == 0 else "Politics"
        confidence = abs(decision_scores[i])
        print(f"\nDocument: '{doc}'")
        print(f"Prediction: {pred_class} (decision score: {decision_scores[i]:.4f})")
    
    
    print("SVM module test completed.")
    print("*" * 40)
