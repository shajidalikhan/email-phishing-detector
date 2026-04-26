import os
import argparse
import pandas as pd
import torch
import shutil
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class PhishingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main(args):
    # Determine the project root to ensure relative paths work regardless of execution directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    data_path = args.data_path if args.data_path else os.path.join(project_root, 'data', 'external', 'Phishing_Email.csv')
    model_save_path = args.output_dir if args.output_dir else os.path.join(project_root, 'models', 'bert')
    
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find the dataset at {data_path}")
        return

    # Check for correct columns and drop NaNs
    if 'Email Text' not in df.columns or 'Email Type' not in df.columns:
        print("Error: Expected columns 'Email Text' and 'Email Type' not found in dataset.")
        return
        
    df = df.dropna(subset=['Email Text', 'Email Type'])
    
    # Map labels to integers
    label_mapping = {'Safe Email': 0, 'Phishing Email': 1}
    df['label'] = df['Email Type'].map(label_mapping)
    df = df.dropna(subset=['label']) # Just in case there are unexpected labels
    df['label'] = df['label'].astype(int)
    
    # Use subset for testing if requested
    if args.subset > 0:
        print(f"Using a subset of {args.subset} samples for testing.")
        # Ensure we have a balanced or stratified sample if possible
        if len(df) > args.subset:
            df = df.sample(n=args.subset, random_state=42)
    
    print(f"Total samples: {len(df)}")
    
    # Split the dataset
    texts = df['Email Text'].tolist()
    labels = df['label'].tolist()
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print("Tokenizing texts...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = PhishingDataset(train_encodings, train_labels)
    val_dataset = PhishingDataset(val_encodings, val_labels)
    
    print("Loading BERT model...")
    # Load the pre-trained model with 2 labels for binary classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # Set up training arguments
    # Trainer automatically uses GPU/MPS if available
    training_args = TrainingArguments(
        output_dir=os.path.join(project_root, 'models', 'bert', 'results'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(project_root, 'models', 'bert', 'logs'),
        logging_steps=10,
        eval_strategy="epoch",  # Changed from evaluation_strategy which is deprecated
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")
    
    # Save the final model and tokenizer
    print(f"Saving final model and tokenizer to {model_save_path}")
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Compress the model directory into a zip file for easy downloading from Colab
    zip_path = os.path.join(project_root, 'models', 'bert_model_export')
    print(f"Compressing the model into {zip_path}.zip for easy download...")
    shutil.make_archive(zip_path, 'zip', model_save_path)
    print("Done! You can now download the bert_model_export.zip file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a BERT model on the phishing dataset.")
    parser.add_argument('--data_path', type=str, default=None, help="Path to the Phishing_Email.csv dataset.")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save the trained model.")
    parser.add_argument('--subset', type=int, default=0, help="Number of samples to use for testing. 0 means use all data.")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size per device.")
    args = parser.parse_args()
    main(args)
