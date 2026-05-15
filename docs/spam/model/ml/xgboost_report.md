# Model Evaluation Report: XGBoost

## 1. Model Details
- **objective**: binary:logistic\n- **base_score**: None\n- **booster**: None\n- **callbacks**: None\n- **colsample_bylevel**: None\n- **colsample_bynode**: None\n- **colsample_bytree**: None\n- **device**: None\n- **early_stopping_rounds**: None\n- **enable_categorical**: False\n- **eval_metric**: logloss\n- **feature_types**: None\n- **feature_weights**: None\n- **gamma**: None\n- **grow_policy**: None\n- **importance_type**: None\n- **interaction_constraints**: None\n- **learning_rate**: None\n- **max_bin**: None\n- **max_cat_threshold**: None\n- **max_cat_to_onehot**: None\n- **max_delta_step**: None\n- **max_depth**: None\n- **max_leaves**: None\n- **min_child_weight**: None\n- **missing**: nan\n- **monotone_constraints**: None\n- **multi_strategy**: None\n- **n_estimators**: None\n- **n_jobs**: -1\n- **num_parallel_tree**: None\n- **random_state**: 42\n- **reg_alpha**: None\n- **reg_lambda**: None\n- **sampling_method**: None\n- **scale_pos_weight**: None\n- **subsample**: None\n- **tree_method**: None\n- **validate_parameters**: None\n- **verbosity**: None\n- **use_label_encoder**: False

## 2. Data Split Details
- **Split Strategy**: 80-20 Stratified Split (random_state=42)\n- **Training Samples**: 64428\n- **Testing Samples**: 16107\n- **Total Dataset Size**: 80535

## 3. Overall Performance Metrics
- **Accuracy**: 0.9886
- **Precision**: 0.9887
- **Recall**: 0.9858
- **F1-Score**: 0.9872

## 4. Confusion Matrix
| | Predicted Ham (0) | Predicted Spam (1) |
|---|---|---|
| **Actual Ham (0)** | 8841 | 81 |
| **Actual Spam (1)** | 102 | 7083 |

## 5. Detailed Classification Report
```text
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      8922
           1       0.99      0.99      0.99      7185

    accuracy                           0.99     16107
   macro avg       0.99      0.99      0.99     16107
weighted avg       0.99      0.99      0.99     16107

```
