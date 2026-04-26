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
