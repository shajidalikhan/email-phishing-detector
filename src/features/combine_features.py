import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def combine_features(tfidf_path, features_csv_path, output_x_path, output_y_path, encoder_path):
    """
    Combines TF-IDF matrix with engineered features and encodes labels.
    """
    print(f"Loading TF-IDF matrix from {tfidf_path}...")
    X_tfidf = np.load(tfidf_path)
    
    print(f"Loading engineered features from {features_csv_path}...")
    df = pd.read_csv(features_csv_path)
    
    # List of engineered feature columns
    engineered_cols = [
        'url_count', 
        'body_length', 
        'special_char_ratio', 
        'has_urgent', 
        'has_verify', 
        'has_password', 
        'has_suspension',
        'has_financial',
        'has_prize'
    ]
    
    print(f"Extracting engineered features: {engineered_cols}...")
    X_engineered = df[engineered_cols].values
    
    # Combine matrices
    print("Combining features...")
    X_final = np.hstack((X_tfidf, X_engineered))
    
    print(f"Final feature matrix shape: {X_final.shape}")
    
    # Encode labels
    print("Encoding labels...")
    le = LabelEncoder()
    y = le.fit_transform(df['Email Type'])
    
    print(f"Classes: {le.classes_}")
    print(f"Label counts: {pd.Series(y).value_counts().to_dict()}")
    
    # Save results
    print(f"Saving final feature matrix to {output_x_path}...")
    np.save(output_x_path, X_final)
    
    print(f"Saving encoded labels to {output_y_path}...")
    np.save(output_y_path, y)
    
    print(f"Saving label encoder to {encoder_path}...")
    joblib.dump(le, encoder_path)
    
    print("Feature combination complete!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    tfidf_path = os.path.join(base_dir, 'models', 'X_tfidf_matrix.npy')
    features_csv_path = os.path.join(base_dir, 'data', 'features_final.csv')
    
    output_x_path = os.path.join(base_dir, 'models', 'X_final.npy')
    output_y_path = os.path.join(base_dir, 'models', 'y.npy')
    encoder_path = os.path.join(base_dir, 'models', 'label_encoder.pkl')
    
    if os.path.exists(tfidf_path) and os.path.exists(features_csv_path):
        combine_features(tfidf_path, features_csv_path, output_x_path, output_y_path, encoder_path)
    else:
        if not os.path.exists(tfidf_path):
            print(f"Error: TF-IDF matrix not found at {tfidf_path}")
        if not os.path.exists(features_csv_path):
            print(f"Error: Features CSV not found at {features_csv_path}")
