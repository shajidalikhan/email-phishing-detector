import os
import sys
import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Add project root to sys.path to handle 'src' imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.features.extract_meta import get_url_count, get_text_length, get_special_char_ratio, get_keyword_flags

# Download necessary NLTK data
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

download_nltk_resources()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_html(text):
    """
    Strips HTML tags from the text.
    """
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ")
    return clean_text

def preprocess_text(text):
    """
    Normalizes text: lowercase, remove punctuation, tokenize, remove stopwords, and lemmatize.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Replace URLs with <URL> token (after counting them in the pipeline)
    text = re.sub(r'http[s]?://\S+', '<URL>', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(clean_tokens)

def build_feature_set(input_path, output_path):
    """
    Main pipeline to load data, engineer features, and save the result.
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Use the 'Email' column
    text_col = 'Email'
    
    if text_col not in df.columns:
        # Fallback or error
        print(f"Error: {text_col} column not found in dataset.")
        return
    
    print("Extracting meta-features...")
    df['url_count'] = df[text_col].apply(get_url_count)
    df['body_length'] = df[text_col].apply(get_text_length)
    df['special_char_ratio'] = df[text_col].apply(get_special_char_ratio)
    
    # Keyword flags
    keyword_flags = df[text_col].apply(get_keyword_flags)
    keyword_df = pd.DataFrame(keyword_flags.tolist())
    df = pd.concat([df, keyword_df], axis=1)
    
    print("Cleaning HTML and preprocessing text...")
    df['cleaned_email'] = df[text_col].apply(clean_html).apply(preprocess_text)
    
    print(f"Saving processed features to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv = os.path.join(base_dir, 'data', 'phishing_dataset_final.csv')
    output_csv = os.path.join(base_dir, 'data', 'features_final.csv')
    
    if os.path.exists(input_csv):
        build_feature_set(input_csv, output_csv)
    else:
        print(f"Input file not found at {input_csv}")
