"""
TEXT PREPROCESSING MODULE

Cleaning a text and preprocessing it for feature extraction.
"""

import re


def text_cleaning(text):
    """
    cleaning and preprocessing a text document.
    
    Multiple stages for :
    
    1. LOWERCASE CONVERSION :
       -> Converting all text to lowercase
       -> Ensures "Sport" and "sport" are treated identically
       -> Reducing vocabulary size
    
    2. URL REMOVAL:
       -> Removing http://, https://, and www. links
       -> URLs don't contribute to classification
    
    3. EMAIL REMOVAL:
       -> Removing email addresses 
       -> Prevents overfitting on sender information
    
    4. SPECIAL CHARACTER REMOVAL:
       -> Keeps only letter and spaces
       -> Removes numbers, punctuation, symbols
       -> Reduces noise in the data
    
    5. WHITESPACE NORMALIZATION:
       -> Replacing multiple spaces with single space
       -> Removing leading/trailing whitespace
       -> Ensuring consistent formatting
    
    Args:
        text : Raw text document
        
    Returns:
        str : Cleaned and normalized text
        
    Example:
        >>> raw --> "Check out https://example.com! Great Article #1!!!"
        >>> clean --> text_cleaning(raw)
        >>> print(clean) --> 'check out great article'
    """
    # Handling non-string inputs 
    if not isinstance(text, str):
        return ""
    
    # 1. Converting to lowercase
    # This ensures case insensitive matching
    text = text.lower()
    
    # 2. Removing URLs
    # Pattern matches http://, https://, and www. links
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 3. Removing email addresses
    # Pattern matches standard email format: user@domain.com
    text = re.sub(r'\S+@\S+', '', text)
    
    # 4. Removing special characters
    # Removing all numbers, punctuation, and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 5. Normalizing whitespace
    # Replacing multiple spaces/tabs/newlines with single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_dataset(X_raw):
    """
    cleaning a text to call docs in dataset.
    
    it is processing entire list of documents by the using of text_cleaning function.
    processing large size of documents
    
    Args:
        X_raw : List of raw text documents
        
    Returns:
        list : List of cleaned text documents
        
    
    """
    print(f"Preprocessing {len(X_raw)} documents...")
    print("Applying : lowercase, URL removal, email removal, special character removal")
    
    # Applying cleaning function to each document
    # Lists comprehension is efficient for this operation
    cleaned_docs = [text_cleaning(doc) for doc in X_raw]
    
    # It is Displaying a sample to show the cleaning effect
    if len(X_raw) > 0:
        print(f"\nSample preprocessing:")
        print(f"Before: {X_raw[0][:100]}...")
        print(f"After:  {cleaned_docs[0][:100]}...")
    
    print(f"Preprocessing complete: {len(cleaned_docs)} documents cleaned")
    print()
    
    return cleaned_docs


def text_statistics(documents):
    """
    statistics calculation about text document collection
    
    providing overview of dataset characteristics:
    -> Average document length
    -> Vocabulary size
    -> Total word count
    
    Args:
        documents : List of text documents
        
    Returns:
        dict : Dictionary containing text statistics
    """
    total_words = 0
    vocabulary = set()
    
    for doc in documents:
        words = doc.split()
        total_words += len(words)
        vocabulary.update(words)
    
    avg_length = total_words / len(documents) if documents else 0
    
    stats = {
        'num_documents': len(documents),
        'total_words': total_words,
        'vocabulary_size': len(vocabulary),
        'avg_doc_length': avg_length
    }
    
    return stats


if __name__ == "__main__":
    # Test the preprocessing module
    print("Testing text preprocessing module...\n")
    
    # Test cases
    test_documents = [
        "Sania Mirza became the first Indian woman to win a WTA singles title at Hyderabad Open! Contact: sports@bbc.com",
        "The 18-year-old MIRZA defeated Ukraine's Bondarenko 6-4, 5-7, 6-3!!! #tennis #India",
        "UK welcomed India-Pakistan decision to open bus link across Kashmir ceasefire line... Visit www.bbc.co.uk for details."
    ]
    
    print("Original documents:")
    for i, doc in enumerate(test_documents, 1):
        print(f"{i}. {doc}")
    
    
    cleaned = preprocess_dataset(test_documents)
    
    print("\nCleaned documents:")
    for i, doc in enumerate(cleaned, 1):
        print(f"{i}. {doc}")
    
    print("\n" + "*" * 60)
    stats = text_statistics(cleaned)
    print("\nText Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
