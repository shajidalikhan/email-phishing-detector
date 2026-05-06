# 📧 Email Phishing Detection – Feature Engineering Pipeline

## 🔍 Overview of Work
The core of this implementation consists of a robust **NLP Preprocessing and Feature Engineering Pipeline**. This system bridges the gap between raw, noisy email data and machine learning models by transforming unstructured text into high-quality, numerical feature sets.

Key pillars of the pipeline:
- **End-to-End Preprocessing**: Comprehensive cleaning, normalization, and tokenization of raw email content.
- **Structural Feature Engineering**: Extraction of 9 custom meta-features, including URL counts and keyword triggers, to detect phishing signals.
- **Vectorization**: A persistent TF-IDF pipeline for optimized textual representation.
- **Inference System**: A modular inference framework designed for real-time email analysis.

---

## ⚙️ Features
- **Intelligent Preprocessing**: Automated stripping of HTML noise and normalization of varied email formats using `BeautifulSoup`.
- **Advanced Tokenization**: Leveraging NLTK for high-precision token isolation, stopword removal, and lemmatization.
- **Dynamic URL Masking**: Preserving structural signals by replacing hyperlinks with a `<URL>` token.
- **Hybrid Feature Fusion**: Combining 5,000 TF-IDF textual features with 9 targeted heuristic meta-features.
- **Persistence**: Systematic saving of vectorizers and encoders for consistent deployment.

---

## 🛠️ Key Dependencies
- **Scikit-learn**: Used for `TfidfVectorizer` and `LabelEncoder`.
- **NLTK**: Used for tokenization, stopword lists, and `WordNetLemmatizer`.
- **BeautifulSoup4**: Essential for cleaning raw email HTML content.
- **Pandas & NumPy**: Core engines for data manipulation and matrix operations.

---

## 🔄 Pipeline Overview
```text
Raw Email
    ↓
HTML Cleaning & Preprocessing
    ↓
TF-IDF Vectorization  <——>  Meta-Feature Extraction
    ↓
Feature Combination
    ↓
Final Feature Matrix (Model Input)
```

---

## 🧹 Preprocessing Steps
- **Remove HTML tags** (BeautifulSoup)
- **Convert text to lowercase**
- **Replace URLs** with `<URL>`
- **Tokenize text** (NLTK)
- **Remove stopwords**
- **Apply lemmatization**

---

## 🧠 Feature Engineering

### Meta Features
- **URL Count**: Detects suspicious link density.
- **Text Length**: Captures abnormal email size.
- **Special Character Ratio**: Identifies obfuscation patterns.

### Keyword Flags
Binary indicators for phishing-related terms:
- Urgent
- Verify
- Password
- Suspension
- Financial
- Prize

---

## 🔢 Text Vectorization (TF-IDF)
- `max_features = 5000`
- `ngram_range = (1, 2)` (unigrams + bigrams)

**Captures:**
- Important words (TF-IDF weighting)
- Short phrases like "verify account" or "immediate action"

*Vectorizer is saved to:* `models/vectorizer.pkl`

---

## 🔗 Feature Integration
- Combine TF-IDF features with meta-features using horizontal stacking.
- **Final dataset:**
  - `X_final.npy`: Feature matrix
  - `y.npy`: Labels

---

## 🔮 Inference Pipeline
Supports processing **new, unseen emails** by reusing the preprocessing steps and the trained TF-IDF vectorizer, ensuring consistency with the training pipeline.

### Example Usage
```python
from src.features.inference import load_vectorizer, prepare_inference_features

# Load the saved vectorizer
vectorizer = load_vectorizer()

# Process a new email
email = "Urgent! Verify your account now."
features = prepare_inference_features(email, vectorizer)
```

---

## 📄 File Responsibilities
Each module in `src/features/` is designed for a specific stage of the pipeline:

- **`build_features.py`**: The central orchestrator. It manages the entire flow from raw CSV to final matrices, calling all other modules in the correct sequence.
- **`extract_meta.py`**: Contains the logic for structural heuristics. It calculates character ratios, link counts, and identifies suspicious keyword clusters (Urgent, Verify, etc.).
- **`vectorize_text.py`**: Handles the NLP transformation. It fits the TF-IDF vectorizer on the cleaned text, transforms it, and saves the `vectorizer.pkl` for later use.
- **`combine_features.py`**: Performs "Feature Fusion." It horizontally stacks the textual and structural features and encodes the target labels (`Email Type`) into numerical values.
- **`inference.py`**: The bridge to deployment. It provides a lightweight function to transform a single raw string into the exact feature format required by the model.

---

## 📁 Project Structure
```text
src/
└── features/
    ├── build_features.py
    ├── extract_meta.py
    ├── vectorize_text.py
    ├── combine_features.py
    └── inference.py
models/
└── vectorizer.pkl
data/
└── processed/
    ├── X_final.npy
    └── y.npy
```

---

---

## 🤝 Handover Guide for Modeling Team
The following files in `data/processed/` and `models/` are ready for model training:

1. **`X_final.npy`**: The final feature matrix.
   - **Shape**: `(N_samples, 5009)`
   - **Structure**: First 5,000 columns are TF-IDF features; the last 9 columns are Meta-features.
2. **`y.npy`**: The target labels encoded as integers.
3. **`label_encoder.pkl`**: Use this to map the model's numerical output back to "Phishing" or "Legitimate" strings.

**Next Steps for Modeling:**
- Load the data using `np.load('data/processed/X_final.npy')`.
- Perform a Train/Test split.
- Train classifiers (Random Forest, SVM, or XGBoost) on this combined feature space.

---

## 🔮 Inference Usage Guide
To process a new, unseen email for prediction, use the `prepare_inference_features` function. This ensures that the new email undergoes the exact same transformation as the training data.

### Implementation Example:
```python
import joblib
import numpy as np
from src.features.inference import prepare_inference_features

# 1. Load your trained model and the saved vectorizer
model = joblib.load('models/your_trained_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# 2. Raw email input
new_email = "URGENT: Your account will be suspended! Verify here: http://bit.ly/phish"

# 3. Transform to feature vector
# This returns a (1, 5009) shape array
features = prepare_inference_features(new_email, vectorizer)

# 4. Predict
prediction = model.predict(features)
```

---

## ▶️ Usage Guide

### 1. Execute Full Pipeline
To process the raw dataset and generate all training files:
```bash
python src/features/build_features.py
```

### 2. Standalone Vectorization
To update only the TF-IDF vectorizer without re-extracting meta-features:
```bash
python src/features/vectorize_text.py
```

### 3. Verify Inference
To test the inference logic with a sample email:
```bash
python src/features/inference.py
```

---

## 📦 Output
- `X_final.npy`: Final feature matrix.
- `y.npy`: Target labels.
- `vectorizer.pkl`: Saved TF-IDF model.

---

## 🚀 Future Improvements
- Integrate classification models (SVM, Random Forest, etc.).
- Add deep learning-based embeddings (BERT, etc.).
- Deploy as API (FastAPI / Flask).
- Build frontend interface.
