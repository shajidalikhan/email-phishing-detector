# BERT Experiment Guide (Shajid)

Welcome to your advanced modeling branch! Here, you will experiment with building a Deep Learning model using BERT to see if it outperforms the Scikit-Learn baseline models.

## Pre-requisites
We have updated the `requirements.txt` on this branch. You will need to install the new HuggingFace library:
```bash
pip install -r requirements.txt
```

## Step-by-Step Experiment Plan

### 1. Understanding the Data Input
Unlike Naive Bayes or SVM, BERT **does not** use TF-IDF features. 
Instead of loading Abhijeet's `tfidf_vectorizer.pkl`, you will load the raw `cleaned_text` from `data/processed/cleaned_emails.csv`. BERT needs actual sentences, not word frequencies.

### 2. Tokenization with HuggingFace
You will use the `transformers` library to load a pre-trained tokenizer.
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example usage:
tokens = tokenizer("Please click this link to verify your account", padding='max_length', truncation=True, max_length=128)
```

### 3. Building the Model
You can use `TFBertForSequenceClassification` (TensorFlow) or `BertForSequenceClassification` (PyTorch) to load the pre-trained BERT weights and add a classification head on top.

```python
from transformers import TFAutoModelForSequenceClassification
model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

### 4. Fine-Tuning
Train the model on your tokenized training set. This is computationally heavy. If you don't have a GPU on your local machine, you should highly consider writing your code in `notebooks/04_bert_experiment.ipynb` and running it on **Google Colab** (which offers free GPUs).

### 5. Evaluation
Compare your BERT model's F1-Score against the Logistic Regression and SVM models you built on the `feature/modeling` branch. If it's significantly better, we can merge this experiment into the final project!
