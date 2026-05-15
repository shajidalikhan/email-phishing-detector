# BERT Testing Script
from transformers import BertTokenizer
import tensorflow as tf

def test_bert(model, tokenizer, X_test):
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512, return_tensors='tf')
    predictions = model.predict(dict(test_encodings)).logits
    return tf.argmax(predictions, axis=1).numpy()