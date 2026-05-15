# Model Evaluation Report: Naive Bayes (MultinomialNB)

## 1. Model Details
- **alpha**: 1.0\n- **class_prior**: None\n- **fit_prior**: True\n- **force_alpha**: True

## 2. Data Split Details
- **Split Strategy**: 80-20 Stratified Split (random_state=42)\n- **Training Samples**: 64428\n- **Testing Samples**: 16107\n- **Total Dataset Size**: 80535

## 3. Overall Performance Metrics
- **Accuracy**: 0.9683
- **Precision**: 0.9850
- **Recall**: 0.9434
- **F1-Score**: 0.9637

## 4. Confusion Matrix
| | Predicted Ham (0) | Predicted Spam (1) |
|---|---|---|
| **Actual Ham (0)** | 8819 | 103 |
| **Actual Spam (1)** | 407 | 6778 |

## 5. Detailed Classification Report
```text
              precision    recall  f1-score   support

           0       0.96      0.99      0.97      8922
           1       0.99      0.94      0.96      7185

    accuracy                           0.97     16107
   macro avg       0.97      0.97      0.97     16107
weighted avg       0.97      0.97      0.97     16107

```
