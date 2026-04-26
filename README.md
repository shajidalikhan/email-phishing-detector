# Phishing Email Detection Framework

Email phishing detection framework using NLP to analyze, classify, and extract suspicious patterns from email content.

## Overview
This project detects and classifies phishing emails using NLP techniques.

## Data Sources
- [PhishTank](https://www.phishtank.com/)
- Enron email dataset

## Documentation
- [Directory Structure](docs/DIRECTORY_STRUCTURE.md): Understand the layout of the project.
- [Contributing](CONTRIBUTING.md): Guidelines on how to clone, setup, and work with git branches.
- [Team Roles](docs/TEAM_ROLES.md): Breakdown of the four main contributor roles and how the team should collaborate.

## Getting the Model and Dataset

Because the trained BERT model (~438 MB) and the full dataset are too large to host on GitHub, you have two options to obtain them to work on the project:

### Option 1: Download from Google Drive (Recommended for Testing)
If you only want to test the model or run the evaluation scripts without spending 1.5 hours training it yourself:
1. Download the `bert_model_export.zip` and `Phishing_Email.csv` files from the team's shared Google Drive link: `[Insert Google Drive Link Here]`
2. Extract the model zip file directly into the `models/bert/` directory of your cloned repository.
3. Place the `Phishing_Email.csv` dataset directly into the `data/external/` directory.

### Option 2: Train the Model Yourself (Recommended for Experimentation)
If you want to train the model from scratch or tune its hyperparameters:
1. Place the `Phishing_Email.csv` dataset into the `data/external/` directory.
2. We highly recommend running the training on Google Colab (using a free T4 GPU) which takes ~1.5 hours. You can paste the contents of `src/models/train_bert.py` into a Colab notebook.
3. Once training completes, download the exported model zip and extract it into your local `models/bert/` folder.

## Testing and Evaluating the BERT Model

We have implemented an advanced deep learning approach using HuggingFace's BERT architecture. You can easily test and evaluate the model using our provided scripts.

### Interactive Phishing Tester
The `test_bert.py` script provides an interactive command-line interface. You can paste email text directly into your terminal to instantly see if it is classified as a phishing attack or a safe email, along with the model's confidence percentage.

1. Ensure your virtual environment is activated:
   ```bash
   .\venv\Scripts\Activate.ps1
   ```
2. Run the interactive testing script:
   ```bash
   python test_bert.py
   ```
3. Paste any email content into the prompt and hit Enter!

### Full Dataset Evaluation
To evaluate the model's overall performance on the test dataset and generate a professional Markdown report, use the evaluation pipeline:
```bash
python src/models/evaluate_bert.py --evaluate --eval_samples 1000
```
*(Setting `--eval_samples 0` will evaluate the entire 175k row dataset).*
This will test the model and generate a comprehensive `docs/bert_evaluation_report.md` containing Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.
