# Model Evaluation Report: LSTM (Deep Learning)

## 1. Model Details
- **Model Type**: Sequential LSTM\n- **Max Words**: 10000\n- **Max Length**: 200\n- **Embedding Dim**: 64\n- **LSTM Units**: 64\n- **Dense Units**: 32\n- **Optimizer**: adam

## 2. Data Split Details
- **Split Strategy**: 80-20 Stratified Split (random_state=42)\n- **Training Samples**: 64428\n- **Testing Samples**: 16107\n- **Total Dataset Size**: 80535

## 3. Overall Performance Metrics
- **Accuracy**: 0.9588
- **Precision**: 0.9748
- **Recall**: 0.9317
- **F1-Score**: 0.9527

## 4. Confusion Matrix
| | Predicted Ham (0) | Predicted Spam (1) |
|---|---|---|
| **Actual Ham (0)** | 8749 | 173 |
| **Actual Spam (1)** | 491 | 6694 |

## 5. Detailed Classification Report
```text
              precision    recall  f1-score   support

           0       0.95      0.98      0.96      8922
           1       0.97      0.93      0.95      7185

    accuracy                           0.96     16107
   macro avg       0.96      0.96      0.96     16107
weighted avg       0.96      0.96      0.96     16107

```
