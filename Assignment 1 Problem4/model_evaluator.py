"""
MODEL EVALUATION MODULE


This module handles model evaluation, performance metrics calculation,
and visualization of results including confusion matrices and comparison charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test, model_name, feature_name):
    """
    Evaluate a trained model on test data and compute performance metrics.
    
    EVALUATION METRICS EXPLAINED:
    ----------------------------
    
    1. ACCURACY: (TP + TN) / Total
       - Overall correctness of predictions
       - Simple but can be misleading for imbalanced data
       - Example: 90% accuracy means 90 out of 100 predictions are correct
    
    2. PRECISION: TP / (TP + FP)
       - Accuracy of positive (Politics) predictions
       - "Of all predicted Politics, how many are actually Politics?"
       - High precision = few false alarms
       - Important when false positives are costly
    
    3. RECALL (Sensitivity): TP / (TP + FN)
       - Coverage of actual positive (Politics) cases
       - "Of all actual Politics articles, how many did we find?"
       - High recall = few missed cases
       - Important when false negatives are costly
    
    4. F1-SCORE: 2 × (Precision × Recall) / (Precision + Recall)
       - Harmonic mean of precision and recall
       - Best metric for imbalanced data
       - Balances precision and recall
       - Range: 0 to 1 (higher is better)
    
    CONFUSION MATRIX:
    ----------------
                    Predicted
                    Sport  Politics
    Actual Sport    [TN]    [FP]
           Politics [FN]    [TP]
    
    - TN (True Negative): Correctly predicted Sport
    - FP (False Positive): Sport wrongly predicted as Politics
    - FN (False Negative): Politics wrongly predicted as Sport
    - TP (True Positive): Correctly predicted Politics
    
    Args:
        model: Trained classifier (Naive Bayes, Logistic Regression, or SVM)
        X_test: Test feature matrix
        y_test: True test labels (0 for Sport, 1 for Politics)
        model_name (str): Name of the model for display
        feature_name (str): Name of feature representation for display
        
    Returns:
        dict: Dictionary containing all performance metrics
            - 'model': Model name
            - 'features': Feature type name
            - 'accuracy': Accuracy score
            - 'precision': Precision score
            - 'recall': Recall score
            - 'f1_score': F1 score
            - 'confusion_matrix': Confusion matrix array
    """
    print(f"Evaluating {model_name} with {feature_name} features...")
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    # average='binary' is used for binary classification (Sport vs Politics)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    
    # Display metrics
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print()
    
    return {
        'model': model_name,
        'features': feature_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, model_name, feature_name, save_path=None):
    """
    Create a heatmap visualization of the confusion matrix.
    
    The confusion matrix visualization helps understand:
    - How many predictions were correct (diagonal elements)
    - What types of errors the model makes (off-diagonal elements)
    - Whether the model is biased toward one class
    
    Args:
        cm: Confusion matrix array (2x2 for binary classification)
        model_name (str): Name of the model
        feature_name (str): Name of features
        save_path (str): Optional path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    # Create heatmap with annotations
    # annot=True: show numbers in cells
    # fmt='d': format as integers
    # cmap='Blues': blue color scheme
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Sport', 'Politics'],
        yticklabels=['Sport', 'Politics'],
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(f'Confusion Matrix: {model_name} with {feature_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved confusion matrix to {save_path}")
    
    plt.close()


def compare_results(results):
    """
    Create a comprehensive comparison table of all model-feature combinations.
    
    This function organizes all experimental results into a DataFrame,
    sorts by F1-score (best metric for imbalanced data), and displays
    a formatted comparison table.
    
    Args:
        results (list): List of result dictionaries from evaluate_model
        
    Returns:
        pandas.DataFrame: Comparison table sorted by F1-score
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("=" * 80)
    
    # Convert results to DataFrame for easy comparison
    df = pd.DataFrame(results)
    
    # Sort by F1-score (best metric for imbalanced data)
    df_sorted = df.sort_values('f1_score', ascending=False).copy()
    
    # Create display DataFrame with formatted percentages
    display_df = df_sorted.copy()
    display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x*100:.2f}%")
    display_df['precision'] = display_df['precision'].apply(lambda x: f"{x*100:.2f}%")
    display_df['recall'] = display_df['recall'].apply(lambda x: f"{x*100:.2f}%")
    display_df['f1_score'] = display_df['f1_score'].apply(lambda x: f"{x*100:.2f}%")
    
    # Select columns for display
    display_cols = ['model', 'features', 'accuracy', 'precision', 'recall', 'f1_score']
    if 'training_time' in display_df.columns:
        display_df['training_time'] = display_df['training_time'].apply(lambda x: f"{x:.4f}s")
        display_cols.append('training_time')
    
    print(display_df[display_cols].to_string(index=False))
    print("=" * 80)
    print()
    
    return df_sorted


def plot_performance_comparison(results_df, save_path=None):
    """
    Create a grouped bar chart comparing all models and features.
    
    This visualization shows:
    - Which model performs best overall
    - How different features affect each model
    - Relative performance differences
    
    Args:
        results_df (pandas.DataFrame): DataFrame containing all experiment results
        save_path (str): Optional path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    plot_data = results_df.copy()
    plot_data['f1_score_pct'] = plot_data['f1_score'] * 100
    
    # Create grouped bar chart
    x = np.arange(len(plot_data))
    width = 0.6
    
    bars = plt.bar(x, plot_data['f1_score_pct'], width, color='steelblue', alpha=0.8)
    
    # Customize plot
    plt.xlabel('Model + Feature Combination', fontsize=12, fontweight='bold')
    plt.ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison (F1-Score)', fontsize=14, fontweight='bold')
    plt.xticks(x, [f"{row['model']}\n{row['features']}" for _, row in plot_data.iterrows()],
               rotation=45, ha='right', fontsize=9)
    plt.ylim(85, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved performance comparison to {save_path}")
    
    plt.close()


def generate_evaluation_report(results, output_file='evaluation_report.txt'):
    """
    Generate a detailed text report of all evaluation results.
    
    Args:
        results (list): List of result dictionaries
        output_file (str): Path to save the report
    """
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('f1_score', ascending=False)
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SPORT VS POLITICS CLASSIFIER - EVALUATION REPORT\n")
        f.write("Student: M25CSA031\n")
        f.write("Assignment: NLU Assignment 1 - Problem 4\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("COMPLETE RESULTS TABLE:\n")
        f.write("-" * 80 + "\n")
        for _, row in df_sorted.iterrows():
            f.write(f"\nModel: {row['model']}\n")
            f.write(f"Features: {row['features']}\n")
            f.write(f"Accuracy:  {row['accuracy']:.4f} ({row['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {row['precision']:.4f}\n")
            f.write(f"Recall:    {row['recall']:.4f}\n")
            f.write(f"F1-Score:  {row['f1_score']:.4f}\n")
            if 'training_time' in row:
                f.write(f"Training Time: {row['training_time']:.4f} seconds\n")
            f.write("-" * 80 + "\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("BEST PERFORMING CONFIGURATION:\n")
        f.write("=" * 80 + "\n")
        best = df_sorted.iloc[0]
        f.write(f"Model:       {best['model']}\n")
        f.write(f"Features:    {best['features']}\n")
        f.write(f"Accuracy:    {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)\n")
        f.write(f"Precision:   {best['precision']:.4f}\n")
        f.write(f"Recall:      {best['recall']:.4f}\n")
        f.write(f"F1-Score:    {best['f1_score']:.4f}\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Evaluation report saved to {output_file}")


if __name__ == "__main__":
    # Test the evaluation module
    print("Testing model evaluation module...\n")
    
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    
    # Sample data
    train_docs = ["baseball game", "hockey match", "president speech", "senate vote"]
    test_docs = ["football game", "political debate"]
    train_labels = np.array([0, 0, 1, 1])
    test_labels = np.array([0, 1])
    
    # Train a simple model
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_docs)
    X_test = vectorizer.transform(test_docs)
    
    model = MultinomialNB()
    model.fit(X_train, train_labels)
    
    # Evaluate
    results = evaluate_model(model, X_test, test_labels, "Naive Bayes", "Bag of Words")
    
    print("Evaluation complete!")
    print(f"F1-Score: {results['f1_score']:.4f}")
