import os
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def save_evaluation_report(model_name, y_true, y_pred, hyperparams, split_details, output_dir, file_name):
    """
    Evaluates predictions and saves a beautifully formatted markdown report.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, file_name)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred)
    
    # Format hyperparams
    hp_str = "\\n".join([f"- **{k}**: {v}" for k, v in hyperparams.items()])
    
    # Format split details
    split_str = "\\n".join([f"- **{k}**: {v}" for k, v in split_details.items()])
    
    md_content = f"""# Model Evaluation Report: {model_name}

## 1. Model Details
{hp_str}

## 2. Data Split Details
{split_str}

## 3. Overall Performance Metrics
- **Accuracy**: {acc:.4f}
- **Precision**: {prec:.4f}
- **Recall**: {rec:.4f}
- **F1-Score**: {f1:.4f}

## 4. Confusion Matrix
| | Predicted Ham (0) | Predicted Spam (1) |
|---|---|---|
| **Actual Ham (0)** | {cm[0][0]} | {cm[0][1]} |
| **Actual Spam (1)** | {cm[1][0]} | {cm[1][1]} |

## 5. Detailed Classification Report
```text
{clf_report}
```
"""
    
    with open(report_path, "w") as f:
        f.write(md_content)
    
    print(f"Evaluation report successfully saved to {report_path}")
