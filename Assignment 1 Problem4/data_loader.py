"""
DATA LOADER MODULE




This part is loading a sports and politics classes from the bbc dataset.

"""

import numpy as np
import os
from sklearn.model_selection import train_test_split


def loading_data(datadirectory='bbc', test_ratio=0.20):
    """
    LOading a data in this function.
    
    Loading a 2 classes from the daaset:
    1.sports with label == 0
    2.politics with label == 1
    

    Args:
        datadirectory : Directory containing BBC dataset 
        test_ratio : Proportion of data to use for testing 
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
            - X_train : Training documents 
            - X_test : Test documents 
            - y_train : Training labels 
            - y_test : Test labels 
    
    Examples :
        >>> X_train, X_test, y_train, y_test = loading_data()
        >>> print(f"Training samples: {len(X_train)}")
        >>> print(f"Sport samples: {sum(y_train == 0)}")
        >>> print(f"Politics samples: {sum(y_train == 1)}")
    """
    
    print("Loading datasets.....")
    print("*" * 40)
    
    # checking dataset downloaded or not
    if not os.path.exists(datadirectory):
        raise FileNotFoundError(
            f"dataset not found. "
            f"Please ensure the dataset is extracted in the '{datadirectory}' directory."
        )
    
    # Initialize lists for documents and labels
    documents = []
    labels = []
    
    print(f"\n Loading articles....")
    print("Loading classes : Sport and Politics\n")
    
    # Load Sport articles with label == 0 
    sportdirectory = os.path.join(datadirectory, 'sport')
    if os.path.exists(sportdirectory):
        sportfiles = [f for f in os.listdir(sportdirectory) if f.endswith('.txt')]
        print(f"Loading {len(sportfiles)} sport articles...")
        
        for filename in sportfiles:

            filepath = os.path.join(sportdirectory, filename)

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:

                    content = f.read()
                    documents.append(content)
                    labels.append(0)  

            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")

    else:
        raise FileNotFoundError(f"Sport directory not found: {sportdirectory}")
    
    # Load Politics articles with label == 1
    politicsdirectory = os.path.join(datadirectory, 'politics')
    if os.path.exists(politicsdirectory):
        politicsfiles = [f for f in os.listdir(politicsdirectory) if f.endswith('.txt')]
        print(f"Loading {len(politicsfiles)} politics articles...")
        
        for filename in politicsfiles:
            filepath = os.path.join(politicsdirectory, filename)

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:

                    content = f.read()
                    documents.append(content)
                    labels.append(1)  

            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")
    else:
        raise FileNotFoundError(f"Politics directory not found: {politicsdirectory}")
    
    if len(documents) == 0:
        raise ValueError("No documents loaded from BBC dataset!")
    
    # converting to numpy array
    labels = np.array(labels)
    
    # Splittting train and test sets
    
    # random_state=42 for reproducibility
    print(f"\nSplitting data: {int((1-test_ratio)*100)}% train, {int(test_ratio*100)}% test")


    X_train, X_test, y_train, y_test = train_test_split(
        documents, labels,
        test_size=test_ratio,
        random_state=42,
        stratify=labels
    )
    
    # Display dataset statistics
    
    print("DATASET STATISTICS  BINARY CLASSIFICATION")
    print("*" * 80)

    print(f"\nTotal documents loaded: {len(documents):,}")

    print(f"  Class 0 for Sport :    {sum(labels == 0):,} articles ({sum(labels == 0)/len(labels)*100:.1f}%)")
    print(f"  Class 1 for Politics: {sum(labels == 1):,} articles ({sum(labels == 1)/len(labels)*100:.1f}%)")
    
    print(f"\nTraining Set ({len(X_train):,} samples):")

    print(f"  Sport:    {sum(y_train == 0):,} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"  Politics: {sum(y_train == 1):,} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")
    
    print(f"\nTest Set ({len(X_test):,} samples):")

    print(f"  Sport:    {sum(y_test == 0):,} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
    print(f"  Politics: {sum(y_test == 1):,} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")

    print("-" * 80)
    print()
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":


    # Test the data loader module
    print("Testing data loader module...\n")
    
    
    print("TESTING : LOADING DATASET ")
    print("*" * 40)
    
    try:
        X_train, X_test, y_train, y_test = loading_data()
        
        print("\nSample Sport article:")

        print("-" * 30)
        sport_idx = np.where(y_train == 0)[0][0]
        print(X_train[sport_idx][:200] + "...")
        
        print("\nSample Politics article:")
        
        print("-" * 50)
        politics_idx = np.where(y_train == 1)[0][0]
        print(X_train[politics_idx][:200] + "...")
        
        
        print(" Data loader testing complete.")
      
        print("*" * 45)

    # exception error    
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("\nPlease ensure the BBC dataset is in the 'bbc' directory.")
