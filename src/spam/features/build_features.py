import os
import sys
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import joblib

# Increase the CSV field size limit to handle potentially very long strings
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))

RAW_DIR = os.path.join(project_root, "data", "spam", "raw")
PROCESSED_DIR = os.path.join(project_root, "data", "spam", "processed")
MODELS_DIR = os.path.join(project_root, "models", "spam")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_standardize():
    print("Loading datasets...")
    
    # 1. emails.csv
    df1_path = os.path.join(RAW_DIR, "emails.csv")
    df1 = pd.DataFrame()
    if os.path.exists(df1_path):
        df1 = pd.read_csv(df1_path)
        # It already has 'text' and 'spam' columns
        df1 = df1[["text", "spam"]]
        print(f"emails.csv loaded: {len(df1)} rows.")
    else:
        print(f"Warning: {df1_path} not found.")

    # 2. enron_spam_data.csv
    df2_path = os.path.join(RAW_DIR, "enron_spam_data.csv")
    df2 = pd.DataFrame()
    if os.path.exists(df2_path):
        df2 = pd.read_csv(df2_path)
        # Rename 'Message' to 'text', 'Spam/Ham' to 'spam'
        df2 = df2.rename(columns={'Message': 'text', 'Spam/Ham': 'spam'})
        df2['spam'] = df2['spam'].map({'spam': 1, 'ham': 0})
        df2 = df2[["text", "spam"]]
        print(f"enron_spam_data.csv loaded: {len(df2)} rows.")
    else:
        print(f"Warning: {df2_path} not found.")

    # 3. processed_data.csv
    df3_path = os.path.join(RAW_DIR, "processed_data.csv")
    df3 = pd.DataFrame()
    if os.path.exists(df3_path):
        # Read using python engine and skip bad lines as in temp.ipynb
        df3 = pd.read_csv(df3_path, engine='python', on_bad_lines='skip')
        # Rename 'message' to 'text', 'label' to 'spam'
        df3 = df3.rename(columns={'message': 'text', 'label': 'spam'})
        df3 = df3[["text", "spam"]]
        print(f"processed_data.csv loaded: {len(df3)} rows.")
    else:
        print(f"Warning: {df3_path} not found.")

    # Combine datasets
    print("Combining datasets...")
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Drop NAs on 'text' and remove duplicates
    initial_len = len(combined_df)
    combined_df = combined_df.dropna(subset=['text', 'spam'])
    combined_df = combined_df.drop_duplicates(subset=['text'])
    
    print(f"Combined dataset: {len(combined_df)} rows (dropped {initial_len - len(combined_df)} NAs/duplicates).")
    
    return combined_df

def main():
    # 1. Load and Standardize
    df = load_and_standardize()
    
    # Save combined dataset
    combined_path = os.path.join(PROCESSED_DIR, "combined_data.csv")
    print(f"Saving combined dataset to {combined_path}...")
    df.to_csv(combined_path, index=False)
    
    # 2. Extract Features (TF-IDF)
    print("Extracting TF-IDF features...")
    # Matching max_features=5000 and stop_words='english' from temp.ipynb
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Fit and transform
    X = vectorizer.fit_transform(df['text'])
    y = df['spam'].values
    
    print(f"TF-IDF matrix shape: {X.shape}")
    
    # Save TF-IDF matrix
    matrix_path = os.path.join(PROCESSED_DIR, "tfidf_matrix.npz")
    print(f"Saving TF-IDF matrix to {matrix_path}...")
    scipy.sparse.save_npz(matrix_path, X)
    
    # Save Labels
    labels_path = os.path.join(PROCESSED_DIR, "labels.npy")
    print(f"Saving labels to {labels_path}...")
    np.save(labels_path, y)
    
    # Save Vectorizer
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    print(f"Saving vectorizer to {vectorizer_path}...")
    joblib.dump(vectorizer, vectorizer_path)
    
    print("Spam feature extraction completed successfully!")

if __name__ == "__main__":
    main()
