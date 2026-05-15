# Naive Bayes Evaluation Script
from sklearn.metrics import classification_report, accuracy_score

def evaluate_naive_bayes(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))