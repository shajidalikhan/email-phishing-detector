import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
    
    # Paths
    processed_dir = os.path.join(project_root, "data", "spam", "processed")
    models_dir = os.path.join(project_root, "models", "spam", "dl")
    os.makedirs(models_dir, exist_ok=True)
    
    data_path = os.path.join(processed_dir, "combined_data.csv")
    
    print("Loading raw text data for LSTM...")
    df = pd.read_csv(data_path)
    
    # Ensure no NaNs
    df = df.dropna(subset=['text', 'spam'])
    X = df['text'].astype(str).tolist()
    y = df['spam'].astype(int).tolist()
    
    print("Performing stratified 80-20 train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Tokenizing text data...")
    max_words = 10000
    max_len = 200
    
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
    
    print("Building LSTM model...")
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print("Training LSTM model (use a GPU for better performance)...")
    model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.1)
    
    # Save the model
    model_path = os.path.join(models_dir, "lstm_model.keras")
    print(f"Saving LSTM model to {model_path}...")
    model.save(model_path)
    
    # Save the tokenizer
    tokenizer_path = os.path.join(models_dir, "lstm_tokenizer.pkl")
    print(f"Saving tokenizer to {tokenizer_path}...")
    joblib.dump(tokenizer, tokenizer_path)
    
    # Save test set for evaluation
    test_features_path = os.path.join(processed_dir, "X_test_lstm.npy")
    test_labels_path = os.path.join(processed_dir, "y_test_lstm.npy")
    np.save(test_features_path, X_test_pad)
    np.save(test_labels_path, y_test)
    
    print("LSTM training complete!")

if __name__ == "__main__":
    main()