# BERT Evaluation Script
from sklearn.metrics import classification_report, accuracy_score

def evaluate_bert(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))