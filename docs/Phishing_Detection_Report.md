# Phishing Email Detection — Classical ML Experiment Guide

Welcome to the phishing detection modeling branch! Here, you will find a complete walkthrough of building a text-based phishing email classifier using **TF-IDF + Classical ML** as the baseline. This guide covers data preparation, model training, evaluation, and inference.

---

## Pre-requisites

Install all required libraries before starting:

```bash
pip install pandas numpy scikit-learn joblib
```

---

## Step-by-Step Experiment Plan

### 1. Understanding the Dataset

The dataset used is `Phishing_Email.csv`. It contains **raw email body text** as the only feature — no URLs, headers, or metadata are used.

| Attribute | Value |
|---|---|
| Total Samples (after cleaning) | 12,469 |
| Safe Email (Label 0) | 7,577 (60.7%) |
| Phishing Email (Label 1) | 4,906 (39.3%) |
| Missing Labels Dropped | 14 |
| Feature Used | Email Text (raw body) |

The dataset is split using a **stratified 60/20/20 strategy** to preserve class proportions across all subsets:

| Split | Size | Purpose |
|---|---|---|
| Training (60%) | ~7,481 samples | Model training only |
| Validation (20%) | ~2,494 samples | Hyperparameter tuning |
| Test (20%) | ~2,497 samples | Final unbiased evaluation (used once) |

---

### 2. Text Vectorization with TF-IDF

Unlike deep learning models, classical ML cannot consume raw text. We convert it to a sparse numerical matrix using **TF-IDF**:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words='english',     # removes filler words (the, is, and...)
    max_features=10000,       # retains 10,000 most informative terms
    ngram_range=(1, 2)        # captures unigrams + bigrams (e.g., "click here")
)
```

> **Note:** The vectorizer is fitted **only on training data** to prevent data leakage. Bigrams significantly improve recall by capturing phrases like `"click here"`, `"verify account"`, `"urgent action"`.

---

### 3. Building the Pipeline

Each model is wrapped in a scikit-learn `Pipeline` that chains vectorization + classification in a single call:

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english',
                               max_features=10000,
                               ngram_range=(1, 2))),
    ('clf', LinearSVC(C=1, random_state=42, max_iter=2000))
])

model.fit(X_train, y_train)       # fit on train only
model.predict(X_test)             # vectorize + classify in one call
```

Three model families were evaluated:

| Model | Notes |
|---|---|
| Logistic Regression | Linear probabilistic classifier, L2 regularization, strong baseline |
| Naive Bayes (MultinomialNB) | Fast, assumes feature independence, works on sparse TF-IDF matrices |
| LinearSVC | Maximum-margin hyperplane, highly effective in high-dimensional text spaces |

---

### 4. Hyperparameter Tuning

`GridSearchCV` with 3-fold CV was run on the SVM pipeline, optimizing for **F1 score**:

```python
from sklearn.model_selection import GridSearchCV

svm_params = {
    'tfidf__max_features': [3000, 5000, 10000],
    'clf__C'             : [0.1, 1, 10]
}

# Best result: C=1, max_features=10000
```

---

### 5. Model Performance — Validation Set

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 96.28% | 94.14% | 96.53% | 95.32% |
| Naive Bayes | 94.91% | 95.91% | 90.93% | 93.35% |
| SVM (Linear, default) | 96.60% | 94.18% | 97.35% | 95.74% |
| **SVM (C=1, 10k feat.) ✅** | **97.00%** | **95.00%** | **98.00%** | **96.00%** |

> ✅ Best model selected based on F1 Score.

---

### 6. Final Test Set Evaluation (Best Model: SVM)

Evaluated **once** on the held-out test set of **2,497 emails**:

| Metric | Safe Email | Phishing Email | Overall |
|---|---|---|---|
| Precision | 0.98 | 0.95 | 0.97 |
| Recall | 0.97 | 0.98 | 0.97 |
| F1 Score | 0.98 | 0.96 | 0.97 |
| Support | 1,516 | 981 | 2,497 |

**Confusion Matrix:**

|  | Predicted: Safe | Predicted: Phishing |
|---|---|---|
| **Actual: Safe** | 1472 | 44 |
| **Actual: Phishing** | 20 | 961 |

- **False Negatives (20):** Phishing emails incorrectly marked safe — the more dangerous error
- **False Positives (44):** Safe emails incorrectly flagged as phishing

---

### 7. Saving & Inspecting the Model

```python
import joblib
import numpy as np

# Save
joblib.dump(model, 'phishing_model.pkl')
print('Model saved to phishing_model.pkl')

# Inspect learned weights
model  = joblib.load('phishing_model.pkl')
tfidf  = model.named_steps['tfidf']
svm    = model.named_steps['clf']
vocab  = tfidf.get_feature_names_out()
weights = svm.coef_[0]

# Top 10 phishing keywords
top_phish = vocab[np.argsort(weights)[-10:]]
print('Phishing indicators:', top_phish)

# Top 10 safe keywords
top_safe = vocab[np.argsort(weights)[:10]]
print('Safe indicators:', top_safe)
```

---

### 8. Running Inference

#### Single Email

```python
import joblib

model = joblib.load('phishing_model.pkl')

def check_email(email_text: str) -> dict:
    pred       = model.predict([email_text])[0]
    score      = model.decision_function([email_text])[0]
    label      = 'PHISHING' if pred == 1 else 'SAFE'
    confidence = 'High' if abs(score) > 1.5 else 'Medium' if abs(score) > 0.5 else 'Low'
    return {'label': label, 'raw_score': round(float(score), 4), 'confidence': confidence}
```

**Expected Output:**

```
[SAFE    ] Score: -1.842  Confidence: High
           Text: Hi John, attached is the Q2 report. Let me know your thoughts...

[PHISHING] Score: +2.641  Confidence: High
           Text: URGENT: Your account has been compromised! Click here NOW...
```

**Score Interpretation:**

| Score Range | Predicted Label | Confidence |
|---|---|---|
| score > +1.5 | Phishing | High |
| 0 < score < +1.5 | Phishing | Low–Medium |
| -1.5 < score < 0 | Safe | Low–Medium |
| score < -1.5 | Safe | High |

#### Batch Processing (Full Inbox / CSV)

```python
import pandas as pd, joblib

model = joblib.load('phishing_model.pkl')

df = pd.read_csv('new_emails.csv')           # must have 'email_text' column
df['prediction'] = model.predict(df['email_text'])
df['score']      = model.decision_function(df['email_text'])
df['label']      = df['prediction'].map({0: 'Safe', 1: 'Phishing'})

df[['email_text', 'label', 'score']].to_csv('results.csv', index=False)
print(df['label'].value_counts())
```

---


## Summary

| Item | Value |
|---|---|
| Best Model | LinearSVC (C=1, max_features=10,000, bigrams) |
| Test Accuracy | 97.00% |
| Test F1 Score (Phishing) | 96.00% |
| Phishing Recall | 98% (catches 98 out of 100 phishing emails) |
| Inference | Single `predict()` call, no GPU needed |
| Saved Model Size | ~few MB (.pkl via joblib) |

The model is lightweight, fast, and production-deployable. A single `predict()` call handles end-to-end inference including text preprocessing. Compare these results against the BERT experiment branch to decide which approach to finalize!
