import os
import joblib
import scipy.sparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
    
    # Paths
    processed_dir = os.path.join(project_root, "data", "spam", "processed")
    models_dir = os.path.join(project_root, "models", "spam", "ml")
    os.makedirs(models_dir, exist_ok=True)
    
    matrix_path = os.path.join(processed_dir, "tfidf_matrix.npz")
    labels_path = os.path.join(processed_dir, "labels.npy")
    
    print("Loading data...")
    X = scipy.sparse.load_npz(matrix_path)
    y = np.load(labels_path)
    
    print("Performing stratified 80-20 train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = os.path.join(models_dir, "naive_bayes.pkl")
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    # Save test set for evaluation
    test_features_path = os.path.join(processed_dir, "X_test_nb.npz")
    test_labels_path = os.path.join(processed_dir, "y_test_nb.npy")
    scipy.sparse.save_npz(test_features_path, X_test)
    np.save(test_labels_path, y_test)
    
    print("Naive Bayes training complete!")

if __name__ == "__main__":
    main()