import os
import sys
import pandas as aluminum
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def vectorize_data(input_csv, vectorizer_path, matrix_path):
    """
    Loads the processed CSV, performs TF-IDF vectorization, and saves the results.
    """
    import pandas as pd
    print(f"Loading processed features from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Ensure there are no NaNs in cleaned_email
    df['cleaned_email'] = df['cleaned_email'].fillna('')
    
    print("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df['cleaned_email'])
    
    print(f"TF-IDF Matrix Shape: {X_tfidf.shape}")
    
    # Save the vectorizer
    print(f"Saving vectorizer to {vectorizer_path}...")
    joblib.dump(tfidf, vectorizer_path)
    
    # Save the matrix as NumPy
    print(f"Saving TF-IDF matrix to {matrix_path}...")
    np.save(matrix_path, X_tfidf.toarray())
    
    # Save the matrix as CSV (with feature names as headers)
    csv_matrix_path = matrix_path.replace('.npy', '.csv')
    print(f"Saving TF-IDF matrix to {csv_matrix_path}...")
    feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=feature_names)
    tfidf_df.to_csv(csv_matrix_path, index=False)
    
    print("Vectorization complete!")

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    input_csv = os.path.join(processed_dir, 'features_final.csv')
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    matrix_path = os.path.join(processed_dir, 'X_tfidf_matrix.npy')
    
    if os.path.exists(input_csv):
        vectorize_data(input_csv, vectorizer_path, matrix_path)
    else:
        print(f"Input file not found at {input_csv}. Please run build_features.py first.")
