# BERT Model Evaluation Report
*Generated on: 2026-04-26 21:24:08*

## 1. Model Details & Parameters
* **Architecture:** `BertForSequenceClassification`
* **Vocabulary Size:** `30522`
* **Max Sequence Length Capacity:** `512`
* **Number of Labels:** `2`
* **Model Path:** `D:\Study Material\Computer Science and Application\M.Tech\Semester 2\NLP\project\email-phishing-detector\models\bert`

## 2. Evaluation Dataset
* **Source:** `D:\Study Material\Computer Science and Application\M.Tech\Semester 2\NLP\project\email-phishing-detector\data\external\Phishing_Email.csv`
* **Samples Evaluated:** `1000`

## 3. Performance Metrics
| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.9790 |
| **Precision** | 0.9658 |
| **Recall** | 0.9826 |
| **F1-Score** | 0.9741 |

## 4. Confusion Matrix

| | Predicted Safe (0) | Predicted Phishing (1) |
| :--- | :--- | :--- |
| **Actual Safe (0)** | 584 (True Negatives) | 14 (False Positives) |
| **Actual Phishing (1)** | 7 (False Negatives) | 395 (True Positives) |

