"""
LOGISTIC REGRESSION CLASSIFIER MODULE


In this section implmenting logistic regression classifier for sports and politics classification.,
logistic regression is a model which estimates class probabilities using the logistic function.

"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression


def logistic_regression(X_train, y_train, C=1.0, max_iter=1000):
    """
    Training a Logistic Regression classifier.
    
    
    
    Args:
        X_train : Training feature matrix (sparse matrix from vectorizer)
        y_train : Training labels (0 for Sport, 1 for Politics)
        C : Inverse of regularization strength (default=1.0)
        max_iter : Maximum number of iterations (default=1000)
        
    Returns:
        tuple : (trained_model, training_time)
            - trained_model : Fitted LogisticRegression classifier
            - training_time : Time taken to train in seconds
            

    """
    
    print("TRAINING LOGISTIC REGRESSION CLASSIFIER")
    print("*" * 80)
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Features: {X_train.shape[1]:,}")
    print(f"Regularization parameter (C): {C}")
    print(f"Maximum iterations: {max_iter}")
    print()
    
    # Recording time
    start_time = time.time()
    
    # Initializing Logistic Regression classifier

    # C = 1.0 : inverse regularization strength 
    # max_iter = 1000 : ensures convergence for large datasets
    # random_state = 42 : for reproducibility
    # solver = 'lbfgs' : Limited-memory BFGS optimizer 
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=42,
        solver='lbfgs'
    )
    
    # Training of the model

    # The model learns a:
    # 1. Weight vector w 
    # 2. Bias term b 
    model.fit(X_train, y_train)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Check if model converged
    if not model.n_iter_[0] < max_iter:
        print(" Warning: Model may not have converged. Consider increasing max_iter.")
    else:
        print(f" Model converged in {model.n_iter_[0]} iterations")
    
    print(f" Training completed in {training_time:.4f} seconds")
    print(f" Model ready for prediction")
    print()
    
    return model, training_time


def feature_importance(model, feature_names, top_n=20):
    """
    Extracting the most important features for each class.
    
    
    
    Args:
        model : Trained LogisticRegression model
        feature_names : List of feature names
        top_n : Number of top features to show per class
        
    Returns:
        dict : Dictionary with top features for each class
    """
    # Get coefficients for each feature
    coefficients = model.coef_[0]
    
    # Top Sport indicators
    sport_indices = np.argsort(coefficients)[:top_n]
    sport_features = [(feature_names[i], coefficients[i]) for i in sport_indices]
    
    # Top Politics indicators
    politics_indices = np.argsort(coefficients)[-top_n:][::-1]
    politics_features = [(feature_names[i], coefficients[i]) for i in politics_indices]
    
    return {
        'sport_indicators': sport_features,
        'politics_indicators': politics_features,
        'intercept': model.intercept_[0]
    }


if __name__ == "__main__":
    # Test the Logistic Regression module
    print("Testing Logistic Regression classifier module...\n")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    
    # Sample training data
    train_docs = [
        # Sport examples (label 0)
        "british hurdler sarah claxton confident win first major medal european indoor championships madrid",
        "india mobile gaming market expected generate millions tech savvy youth market driving growth",
        "hyundai motor plans build second plant india meet growing demand cars chennai tamil nadu",
        "indian market grow vehicles reaching million driven poor public transport low car ownership",
        
        # Politics examples (label 1)
        "uk welcomed decision india pakistan open bus link ceasefire line dividing disputed kashmir region",
        "foreign secretary jack straw praised spirit cooperation india pakistan achieving breakthrough peace process",
        "india rupee hit five year high standard poor raised country foreign currency rating",
        "indian government cleared proposal allowing foreign direct investment construction sector boost infrastructure"
    ]
    
    train_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])  
    
    # Extracting features
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_docs)
    
    # Training of the model
    model, train_time = logistic_regression(X_train, train_labels)
    
    # Getting feature importance
    feature_names = vectorizer.get_feature_names_out()
    importance = feature_importance(model, feature_names, top_n=5)
    
    
    print("FEATURE IMPORTANCE")
    print("*" * 80)
    print(f"Intercept (bias): {importance['intercept']:.4f}\n")
    
    print("Top Sport indicators:")
    for word, coef in importance['sport_indicators']:
        print(f"  {word:20s}: {coef:+.4f}")
    
    print("\nTop Politics indicators:")
    for word, coef in importance['politics_indicators']:
        print(f"  {word:20s}: {coef:+.4f}")
    
    # Test prediction
    test_docs = ["baseball team wins championship", "government announces new policy"]
    X_test = vectorizer.transform(test_docs)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print("\n" )
    print("TEST PREDICTIONS")
    print("*" * 30)
    for i, doc in enumerate(test_docs):
        pred_class = "Sport" if predictions[i] == 0 else "Politics"
        confidence = probabilities[i][predictions[i]] * 100
        print(f"Document: '{doc}'")
        print(f"Prediction: {pred_class} (confidence: {confidence:.2f}%)")
        print(f"Probabilities: Sport={probabilities[i][0]*100:.2f}%, Politics={probabilities[i][1]*100:.2f}%")
        print()
