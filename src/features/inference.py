import os
import sys
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.features.extract_meta import get_url_count, get_text_length, get_special_char_ratio, get_keyword_flags
from src.features.build_features import clean_html, preprocess_text

def load_vectorizer(path=None):
    """Loads the pre-trained TF-IDF vectorizer."""
    if path is None:
        path = os.path.join(project_root, 'models', 'vectorizer.pkl')
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorizer not found at {path}. Please run vectorize_text.py first.")
    
    return joblib.load(path)

def prepare_inference_features(email_text, vectorizer):
    """
    Transforms a raw email string into a feature vector for model inference.
    
    Args:
        email_text (str): Raw email body or content.
        vectorizer: Loaded TF-IDF vectorizer object.
        
    Returns:
        np.ndarray: Combined feature vector of shape (1, N).
    """
    # 1. Extract Meta-features
    url_count = get_url_count(email_text)
    body_length = get_text_length(email_text)
    special_char_ratio = get_special_char_ratio(email_text)
    
    # Keyword flags (returns dict)
    kw_flags = get_keyword_flags(email_text)
    
    # Meta-features in the exact order as combine_features.py
    meta_features = [
        url_count,
        body_length,
        special_char_ratio,
        kw_flags['has_urgent'],
        kw_flags['has_verify'],
        kw_flags['has_password'],
        kw_flags['has_suspension'],
        kw_flags['has_financial'],
        kw_flags['has_prize']
    ]
    
    X_meta = np.array(meta_features).reshape(1, -1)
    
    # 2. Text Preprocessing
    cleaned_text = clean_html(email_text)
    preprocessed_text = preprocess_text(cleaned_text)
    
    # 3. TF-IDF Transformation
    X_tfidf = vectorizer.transform([preprocessed_text]).toarray()
    
    # 4. Combine Features (Horizontal Stack)
    # Order: [TF-IDF Features, Meta Features] - matching combine_features.py logic
    X_final = np.hstack((X_tfidf, X_meta))
    
    return X_final

if __name__ == "__main__":
    # Example Usage & Verification
    try:
        tfidf_vec = load_vectorizer()
        print("✓ Vectorizer loaded successfully.")
        
        sample_email = """
        Subject: URGENT: Action Required on your account!
        
        Dear User,
        
        We have detected unauthorized access to your account. Please click the link below 
        to verify your identity and avoid account suspension.
        
        http://secure-login-update.com/verify
        
        If you don't act within 24 hours, your account will be locked.
        
        Regards,
        Security Team
        """
        
        print("\nProcessing sample email...")
        features = prepare_inference_features(sample_email, tfidf_vec)
        
        print(f"Final Feature Vector Shape: {features.shape}")
        print(f"First 10 TF-IDF values: {features[0, :10]}")
        print(f"Last 9 Meta-features: {features[0, -9:]}")
        
        # Validation of Meta-features
        expected_meta = features[0, -9:]
        print("\nMeta-feature Validation:")
        print(f"- URL Count: {expected_meta[0]} (Expected: 1)")
        print(f"- Body Length: {expected_meta[1]}")
        print(f"- Has Urgent: {expected_meta[3]} (Expected: 1)")
        print(f"- Has Suspension: {expected_meta[6]} (Expected: 1)")
        
        print("\n✓ Inference pipeline is ready!")
        
    except Exception as e:
        print(f"Error during verification: {e}")
