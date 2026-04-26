# Model Development Planning - Phishing Email Classification

## 1. Objective

The goal of this module is to design, implement, and evaluate machine learning models for detecting phishing emails.

This is formulated as a binary classification problem:
- 0 → Safe Email
- 1 → Phishing Email

The primary focus is to maximize detection of phishing emails while minimizing false negatives.

---

## 2. Dataset Understanding

The dataset consists of:
- Email Text: Raw textual content of emails
- Email Type: Label (Safe / Phishing)

### Key Challenges:
- Noisy and unstructured text data
- Potential class imbalance
- High dimensional sparse feature space after vectorization

---

## 3. Data Preprocessing Strategy

The preprocessing pipeline will include:

### 3.1 Data Cleaning
- Remove null or missing email entries
- Validate label consistency

### 3.2 Label Encoding
- Convert labels into binary format:
  - Safe Email → 0
  - Phishing Email → 1

### 3.3 Text Processing
- Convert text to lowercase
- Remove English stopwords
- Tokenization handled implicitly by TF-IDF

### 3.4 Feature Extraction
- Use TF-IDF Vectorization
- Limit vocabulary size (e.g., 5000–10000 features) to control dimensionality and improve performance

---

## 4. Dataset Splitting Strategy

The dataset will be divided into three subsets:

- Training Set → 70%
- Validation Set → 15%
- Test Set → 15%

### Purpose:
- Training Set: Used to train the models
- Validation Set: Used for hyperparameter tuning and model selection
- Test Set: Used only for final evaluation to ensure unbiased performance estimation

### Important:
Stratified sampling will be used to maintain class distribution across all splits.

---

## 5. Model Selection

The following machine learning models will be implemented:

### 5.1 Logistic Regression
- Efficient and interpretable
- Performs well on linearly separable sparse data

### 5.2 Naive Bayes (Multinomial)
- Strong baseline for text classification
- Computationally efficient

### 5.3 Support Vector Machine (LinearSVC)
- Effective in high-dimensional spaces
- Robust to overfitting in sparse feature settings
- Expected to provide the best performance

---

## 6. Training Strategy

- Use training set (70%) to fit models
- Use validation set (15%) for hyperparameter tuning
- Use scikit-learn pipelines to combine:
  - TF-IDF Vectorizer
  - Classification model

This ensures a clean and reproducible workflow.

---

## 7. Evaluation Metrics

The following metrics will be used:

- Accuracy
- Precision
- Recall
- F1-score

### Key Focus:
Recall is the most critical metric because failing to detect phishing emails (false negatives) poses a high risk.

F1-score will be used as the primary metric for model comparison as it balances precision and recall.

---

## 8. Hyperparameter Tuning Strategy

Hyperparameter tuning will be performed using the validation set.

### Logistic Regression:
- Regularization parameter (C)

### SVM (LinearSVC):
- Regularization parameter (C)

### TF-IDF:
- Maximum number of features

The best hyperparameters will be selected based on validation F1-score.

---

## 9. Model Comparison Plan

All models will be evaluated on the validation set using:
- F1 Score (primary metric)
- Recall (secondary priority)
- Accuracy (supporting metric)

The best-performing model will then be evaluated on the test set for final reporting.

---

## 10. Final Evaluation Strategy

- The selected model will be retrained (if required)
- Final performance will be reported on the test set only
- A detailed classification report will be generated

---

## 11. Expected Outcome

- A robust phishing email detection system
- Comparative analysis of multiple machine learning models
- Identification of the most effective model for deployment

---

## 12. Future Improvements

- Implement deep learning models (Transformers)
- Use pretrained embeddings instead of TF-IDF
- Apply techniques to handle class imbalance (oversampling, SMOTE)
- Deploy model as an API for real-world usage
