# BERT Training Script
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

def train_bert(X_train, y_train, model_name='bert-base-uncased', epochs=3, batch_size=16):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512, return_tensors='tf')
    dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
    model.fit(dataset, epochs=epochs)
    return model, tokenizer