# Model Evaluation Report: Random Forest

## 1. Model Details
- **bootstrap**: True\n- **ccp_alpha**: 0.0\n- **class_weight**: None\n- **criterion**: gini\n- **max_depth**: None\n- **max_features**: sqrt\n- **max_leaf_nodes**: None\n- **max_samples**: None\n- **min_impurity_decrease**: 0.0\n- **min_samples_leaf**: 1\n- **min_samples_split**: 2\n- **min_weight_fraction_leaf**: 0.0\n- **monotonic_cst**: None\n- **n_estimators**: 100\n- **n_jobs**: -1\n- **oob_score**: False\n- **random_state**: 42\n- **verbose**: 0\n- **warm_start**: False

## 2. Data Split Details
- **Split Strategy**: 80-20 Stratified Split (random_state=42)\n- **Training Samples**: 64428\n- **Testing Samples**: 16107\n- **Total Dataset Size**: 80535

## 3. Overall Performance Metrics
- **Accuracy**: 0.9908
- **Precision**: 0.9901
- **Recall**: 0.9893
- **F1-Score**: 0.9897

## 4. Confusion Matrix
| | Predicted Ham (0) | Predicted Spam (1) |
|---|---|---|
| **Actual Ham (0)** | 8851 | 71 |
| **Actual Spam (1)** | 77 | 7108 |

## 5. Detailed Classification Report
```text
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      8922
           1       0.99      0.99      0.99      7185

    accuracy                           0.99     16107
   macro avg       0.99      0.99      0.99     16107
weighted avg       0.99      0.99      0.99     16107

```
