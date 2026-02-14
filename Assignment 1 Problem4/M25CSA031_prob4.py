"""
MAIN EXECUTION 


THe main execution pipeline:
1. Data loading from bbc dataset
2. Preprocessing of text
3. Feature extraction 
4. Training models 
5. Evaluation of models 
6. Compare results and generate visualizations

OUTPUT:
    -> Console output with progress and results
    -> Confusion matrix images 
    -> Performance comparison chart 
    -> Evaluation report 
"""

import warnings
warnings.filterwarnings('ignore')

# Importing custom modules
from data_loader import loading_data
from text_preprocessor import preprocess_dataset
from feature_extractor import (
    bagofword_features,
    tfidf_features,
    ngram_features
)
from naive_bayes_model import naive_bayestraining
from logistic_regression_model import logistic_regression
from svm_model import svm
from model_evaluator import (
    evaluate_model,
    compare_results,
    plot_confusion_matrix,
    plot_performance_comparison,
    generate_evaluation_report
)


def main():
    """
    Main execution pipeline
    
    Overview:
    
    
    1: DATA LOADING
    -> Downloading a dataset
    -> Categories used : sport and politics
    -> Splitting into training and testing sets
    
    2: TEXT PREPROCESSING
    -> Converting to lowercase
    -> Removing URLs, emails, special characters
    -> Normalizing whitespace
    
    3: FEATURE EXTRACTION
    -> Bag of Words : Simple word count representation
    -> TF-IDF : Weighted word frequency representation
    -> N-grams : Captures word sequences and context
    
    4: MODEL TRAINING
    -> Naive Bayes : Fast probabilistic classifier
    -> Logistic Regression : Linear classifier with probabilities
    -> SVM : Maximum margin classifier
    
    5: EVALUATION
    -> Calculating accuracy, precision, recall, F1-score
    -> Generating confusion matrices
    
    6: VISUALIZATION
    -> Confusion matrices for top models
    -> Performance comparison chart
    -> Detailed evaluation report


    """
    print("\n")
    print("SPORT VS POLITICS TEXT CLASSIFIER")

    print("Dataset: BBC NEWS")
    print("*" * 40 + "\n")
    
    # 1 : LOADING DATA
    print("1 : LOADING DATA")
    print("-" * 80)
    
    # Loading BBC News dataset 
    X_train_raw, X_test_raw, y_train, y_test = loading_data()
    
    # 2 : PREPROCESSING TEXT
    print("\n2 : PREPROCESSING TEXT")
    print("-" * 80)
    X_train_clean = preprocess_dataset(X_train_raw)
    X_test_clean = preprocess_dataset(X_test_raw)
    
    # 3 : FEATURE EXTRACTION
    print("\n3 : FEATURE EXTRACTION")
    print("-" * 80)
    print("Extracting features using three different methods...\n")
    
    # 1: Bag of Words
    X_train_bow, X_test_bow, bow_vectorizer = bagofword_features(
        X_train_clean, X_test_clean, max_features=5000
    )
    
    # 2: TF-IDF
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_features(
        X_train_clean, X_test_clean, max_features=5000
    )
    
    # 3: N-grams model
    X_train_ngram, X_test_ngram, ngram_vectorizer = ngram_features(
        X_train_clean, X_test_clean, ngram_range=(1, 2), max_features=5000
    )
    
    # Stored all feature sets for iteration
    feature_sets = [
        (X_train_bow, X_test_bow, "Bag of Words"),
        (X_train_tfidf, X_test_tfidf, "TF-IDF"),
        (X_train_ngram, X_test_ngram, "N-grams (1-2)")
    ]
    
    # 4 & 5 : TRAIN AND EVALUATE ALL COMBINATIONS
    print("\n4 & 5 : MODEL TRAINING AND EVALUATION")
    print("-" * 80)
    print(f"Running {len(feature_sets)} features × 3 models = {len(feature_sets) * 3} experiments\n")
    
    results = []
    experiment_num = 0
    total_experiments = len(feature_sets) * 3
    
    # Print header for progress tracking
    
    print(f"{'Exp':<4} | {'Feature Type':<15} | {'Model':<22} | {'Accuracy':<10} | {'F1-Score':<10} | {'Time':<10}")
    print("*" * 100)
    
    # Iterating through all feature model combinations
    for X_train_feat, X_test_feat, feature_name in feature_sets:
        
        # 1 : Naive Bayes
        experiment_num += 1
        print(f"\n[{experiment_num}/{total_experiments}] Training {feature_name} + Naive Bayes...")
        
        nb_model, nb_time = naive_bayestraining(X_train_feat, y_train)
        nb_results = evaluate_model(nb_model, X_test_feat, y_test, "Naive Bayes", feature_name)
        nb_results['training_time'] = nb_time
        results.append(nb_results)
        
        print(f"{experiment_num:<4} | {feature_name:<15} | {'Naive Bayes':<22} | "
              f"{nb_results['accuracy']:.4f}     | {nb_results['f1_score']:.4f}     | {nb_time:.4f}s")
        
        # 2 : Logistic Regression
        experiment_num += 1
        print(f"\n[{experiment_num}/{total_experiments}] Training {feature_name} + Logistic Regression...")
        
        lr_model, lr_time = logistic_regression(X_train_feat, y_train)
        lr_results = evaluate_model(lr_model, X_test_feat, y_test, "Logistic Regression", feature_name)
        lr_results['training_time'] = lr_time
        results.append(lr_results)
        
        print(f"{experiment_num:<4} | {feature_name:<15} | {'Logistic Regression':<22} | "
              f"{lr_results['accuracy']:.4f}     | {lr_results['f1_score']:.4f}     | {lr_time:.4f}s")
        
        # 3 : SVM
        experiment_num += 1
        print(f"\n[{experiment_num}/{total_experiments}] Training {feature_name} + SVM...")
        
        svm_model, svm_time = svm(X_train_feat, y_train)
        svm_results = evaluate_model(svm_model, X_test_feat, y_test, "SVM", feature_name)
        svm_results['training_time'] = svm_time
        results.append(svm_results)
        
        print(f"{experiment_num:<4} | {feature_name:<15} | {'SVM':<22} | "
              f"{svm_results['accuracy']:.4f}     | {svm_results['f1_score']:.4f}     | {svm_time:.4f}s")
    
    print("*" * 100)
    
    # 6 : COMPARE RESULTS
    print("\n6 : RESULTS ANALYSIS AND VISUALIZATION")
    print("-" * 60)
    
    # Comparing all results
    results_df = compare_results(results)
    
    # best model
    best_result = results[0]  
    
    print("\n")
    print("BEST PERFORMING CONFIGURATION")
    print("-" * 60)
    print(f"Model:        {best_result['model']}")
    print(f"Features:     {best_result['features']}")
    print(f"Accuracy:     {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"Precision:    {best_result['precision']:.4f}")
    print(f"Recall:       {best_result['recall']:.4f}")
    print(f"F1-Score:     {best_result['f1_score']:.4f}")
    print(f"Train Time:   {best_result['training_time']:.4f} seconds")
    print("-" * 60)
    
    # 7 : GENERATE VISUALIZATIONS
    print("\n7 : GENERATE VISUALIZATIONS")
    print("-" * 60)
    
    # Generating confusion matrices for top 3 models
    print("Creating confusion matrices for top 3 models.....")
    top_3_results = results_df.head(3).to_dict('records')
    
    for i, result in enumerate(top_3_results, 1):
        filename = f"confusion_matrix_{result['model'].replace(' ', '_').lower()}_{result['features'].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()}.png"
        plot_confusion_matrix(
            result['confusion_matrix'],
            result['model'],
            result['features'],
            save_path=filename
        )
    
    # Generating performance comparison chart
    print("Creating performance comparison chart.....")
    plot_performance_comparison(results_df, save_path="performance_comparison.png")
    
    # Generating detailed evaluation report
    print("Generating evaluation report.....")
    generate_evaluation_report(results, output_file="evaluation_report.txt")
    
    # FINAL SUMMARY
    print("\n")
    print("CLASSIFICATION PIPELINE COMPLETE")
    print("-" * 60)
    print("\nSaved Files:")
    print(" confusion_matrix_*.png")
    print(" performance_comparison.png")
    print(" evaluation_report.txt")
    print("\nKey Findings:")
    print(f"Best Model: {best_result['model']} with {best_result['features']}")
    print(f"Best F1-Score: {best_result['f1_score']:.4f}")
    print(f"Best Accuracy: {best_result['accuracy']*100:.2f}%")
    print(f"Training Time: {best_result['training_time']:.4f} seconds")
    print("\nEnd of experiments ")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    """
    main function calling block
    """
    main()