# BERT Model Failure Analysis & Edge Case Evaluation

## 1. Model Overview and Specifications

For the advanced deep learning experiment phase of the Email Phishing Detector project, the team utilized a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model. 

### Model Details
* **Base Architecture**: `bert-base-uncased` (12-layer, 768-hidden, 12-heads, 110M parameters).
* **Objective**: Fine-tuned for Sequence Classification (Binary Classification: `Safe Email` vs `Phishing Email`).
* **Input Constraints**: Tokenized with a maximum sequence length of 128 tokens, utilizing padding and truncation.
* **Hardware**: Trained utilizing the HuggingFace `Trainer` API with hardware acceleration (MPS/CUDA).
* **Dataset**: Trained on `Phishing_Email.csv`, comprising approximately 175,000 emails, split 80/20 for training and validation.

---

## 2. The Edge Case Experiment

While the model achieved high accuracy on standard validation data, it was subjected to "Edge Case" testing to evaluate its true contextual understanding. The model systematically failed on highly sophisticated test inputs, revealing critical insights into its learning process.

### Scenario A: False Negatives (Spear Phishing Marked as Safe)
The model was fed highly sophisticated "Spear Phishing" emails. These emails contained no obvious grammatical errors, used polite corporate language, and lacked classic spam keywords like "URGENT" or "WINNER". 

**Example Input:**
> "Hi John, attached is the Q4 forecast document you requested. I uploaded it to our secure SharePoint portal since the file size is too large for email. Please review the highlighted sections before tomorrow's board meeting: http://sharepoint-auth-login.com/view/Q4_forecast"

**Prediction:** `Safe Email`
**Reality:** `Phishing Email` (Malicious domain disguised as SharePoint).

**Why it Failed:** 
The training dataset lacks sufficient examples of sophisticated corporate spear-phishing. Because the vast majority of phishing emails in the dataset feature poor grammar and aggressive language, the model falsely learned that polite, well-structured corporate jargon guarantees safety.

### Scenario B: False Positives (Legitimate Alerts Marked as Phishing)
The model was fed 100% legitimate corporate and banking alerts. These emails contained high urgency, financial warnings, and robotic system-generated language.

**Example Input:**
> "Automated Alert: Your Active Directory password will expire in exactly 2 days. To change your password, please log in to the employee self-service portal at https://sso.mycompany.com/identity/login. If you fail to update it before Friday, your account will be locked out and you will need to contact the Helpdesk."

**Prediction:** `Phishing Email`
**Reality:** `Safe Email` (Standard automated IT notification).

**Why it Failed:**
The model fell victim to **Spurious Correlation**. In the training data, words like `"expire"`, `"locked out"`, and `"unauthorized"` appear almost exclusively in phishing emails. The model learned a simple heuristic ("scary words = phishing") rather than understanding the deeper context of a legitimate system notification.

---

## 3. Conclusions and Next Steps

This failure analysis is a crucial component of the evaluation phase. It proves that while BERT is highly capable, a model is only as good as the diversity of its training data.

