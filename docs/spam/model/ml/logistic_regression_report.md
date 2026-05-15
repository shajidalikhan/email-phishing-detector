# Model Evaluation Report: Logistic Regression

## 1. Model Details
- **C**: 1.0\n- **class_weight**: None\n- **dual**: False\n- **fit_intercept**: True\n- **intercept_scaling**: 1\n- **l1_ratio**: 0.0\n- **max_iter**: 1000\n- **n_jobs**: None\n- **penalty**: deprecated\n- **random_state**: 42\n- **solver**: lbfgs\n- **tol**: 0.0001\n- **verbose**: 0\n- **warm_start**: False

## 2. Data Split Details
- **Split Strategy**: 80-20 Stratified Split (random_state=42)\n- **Training Samples**: 64428\n- **Testing Samples**: 16107\n- **Total Dataset Size**: 80535

## 3. Overall Performance Metrics
- **Accuracy**: 0.9883
- **Precision**: 0.9866
- **Recall**: 0.9872
- **F1-Score**: 0.9869

## 4. Confusion Matrix
| | Predicted Ham (0) | Predicted Spam (1) |
|---|---|---|
| **Actual Ham (0)** | 8826 | 96 |
| **Actual Spam (1)** | 92 | 7093 |

## 5. Detailed Classification Report
```text
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      8922
           1       0.99      0.99      0.99      7185

    accuracy                           0.99     16107
   macro avg       0.99      0.99      0.99     16107
weighted avg       0.99      0.99      0.99     16107

```
