import os
import sys
import numpy as np
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

sys.path.append(os.path.join(project_root, "src", "spam", "models"))
from evaluate_utils import save_evaluation_report

def main():
    processed_dir = os.path.join(project_root, "data", "spam", "processed")
    models_dir = os.path.join(project_root, "models", "spam", "dl")
    docs_dir = os.path.join(project_root, "docs", "spam", "model", "dl")
    
    test_features_path = os.path.join(processed_dir, "X_test_lstm.npy")
    test_labels_path = os.path.join(processed_dir, "y_test_lstm.npy")
    model_path = os.path.join(models_dir, "lstm_model.keras")
    
    print("Loading test data and model...")
    X_test = np.load(test_features_path)
    y_test = np.load(test_labels_path)
    model = tf.keras.models.load_model(model_path)
    
    print("Generating predictions...")
    probs = model.predict(X_test, verbose=1)
    y_pred = (probs >= 0.5).astype(int).flatten()
    
    hyperparams = {
        "Model Type": "Sequential LSTM",
        "Max Words": 10000,
        "Max Length": 200,
        "Embedding Dim": 64,
        "LSTM Units": 64,
        "Dense Units": 32,
        "Optimizer": "adam"
    }
    
    test_size = len(y_test)
    train_size = int(test_size / 0.2 * 0.8)
    
    split_details = {
        "Split Strategy": "80-20 Stratified Split (random_state=42)",
        "Training Samples": train_size,
        "Testing Samples": test_size,
        "Total Dataset Size": train_size + test_size
    }
    
    print("Evaluating and saving report...")
    save_evaluation_report(
        model_name="LSTM (Deep Learning)",
        y_true=y_test,
        y_pred=y_pred,
        hyperparams=hyperparams,
        split_details=split_details,
        output_dir=docs_dir,
        file_name="lstm_report.md"
    )

if __name__ == "__main__":
    main()