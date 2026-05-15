import os
import sys
import joblib
import scipy.sparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

sys.path.append(os.path.join(project_root, "src", "spam", "models"))
from evaluate_utils import save_evaluation_report

def main():
    processed_dir = os.path.join(project_root, "data", "spam", "processed")
    models_dir = os.path.join(project_root, "models", "spam", "ml")
    docs_dir = os.path.join(project_root, "docs", "spam", "model", "ml")
    
    test_features_path = os.path.join(processed_dir, "X_test_rf.npz")
    test_labels_path = os.path.join(processed_dir, "y_test_rf.npy")
    model_path = os.path.join(models_dir, "random_forest.pkl")
    
    print("Loading test data and model...")
    X_test = scipy.sparse.load_npz(test_features_path)
    y_test = np.load(test_labels_path)
    model = joblib.load(model_path)
    
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    
    hyperparams = model.get_params()
    
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
        model_name="Random Forest",
        y_true=y_test,
        y_pred=y_pred,
        hyperparams=hyperparams,
        split_details=split_details,
        output_dir=docs_dir,
        file_name="random_forest_report.md"
    )

if __name__ == "__main__":
    main()