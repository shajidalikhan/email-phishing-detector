# Team Roles and Responsibilities

To ensure clear ownership and parallel progress, this project is divided among four contributors. Below is the breakdown of responsibilities and how the team should collaborate.

---

## 👤 Harshit Ekka: Data Engineer (Data Collection & Management)
**Assigned Branch:** `feature/data-collection`
**Main Responsibility:** Build and manage the dataset

**Tasks:**
* Collect phishing data from PhishTank and optionally the Enron dataset
* Combine phishing + legitimate emails into one dataset
* Label data properly (`phishing = 1`, `legitimate = 0`)
* Handle missing or inconsistent entries
* Store dataset in a structured format (CSV / DataFrame)

**Deliverables:**
* Clean dataset file
* Data documentation (source, size, fields)

---

## 👤 Abhijeet Singh: NLP Preprocessing & Feature Engineering
**Assigned Branch:** `feature/preprocessing`
**Main Responsibility:** Prepare text for modeling

**Tasks:**
* Remove HTML tags, URLs (or extract them separately)
* Lowercasing, punctuation removal
* Tokenization using NLTK
* Stopword removal, stemming/lemmatization
* Convert text into features:
  * TF-IDF
  * Bag of Words
* Extract additional features:
  * Number of links
  * Presence of suspicious words (e.g., “urgent”, “verify”, “password”)

**Deliverables:**
* Preprocessed dataset
* Feature vectors ready for model input

---

## 👤 Shajid & Umang: Model Developers & Evaluators (ML/NLP Model & Evaluation)
**Assigned Branches:** `feature/modeling` (for model training) and `feature/evaluation` (for metrics and demo)
**Main Responsibility:** Build, train, evaluate the classifier, and extract insights.

**Tasks:**
* Implement models using scikit-learn (Logistic Regression, Naive Bayes, SVM)
* Optionally build deep learning model using TensorFlow / Keras
* Train and tune models (hyperparameters)
* Compare multiple models
* Evaluate model (Accuracy, Precision, Recall, F1-score, Confusion Matrix)
* Visualize results using graphs (matplotlib/seaborn)
* Extract insights:
  * Most common phishing keywords
  * Suspicious link patterns
* Build a simple demo (Input email → Output prediction)

**Deliverables:**
* Trained model(s) and performance comparison
* Evaluation report
* Graphs & visualizations
* Final demo / interface

---

## 🔄 How the team should collaborate
* **Harshit Ekka** → gives dataset to **Abhijeet Singh**
* **Abhijeet Singh** → gives processed features to **Shajid and Umang**
* **Shajid and Umang** → validate, report, and create the final demo

---

## 💡 Optional Add-ons
* Use **BERT-based model** for better accuracy
* Build a simple web app (Flask/Streamlit)
* Real-time phishing detection demo
