# 🎣 Phishing Email Detection Framework

## Person 1 — Data Engineer

**Author:** Harshit Ekka  

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-Web%20Crawling-orange)
![Requests](https://img.shields.io/badge/Requests-API%20Integration-yellow)
![Status](https://img.shields.io/badge/Status-In%20Progress-blue)

---

## 📌 Project Overview

Phishing is one of the most common cyberattacks where attackers trick users into revealing sensitive information like passwords, credit card numbers, or personal data by disguising themselves as trusted entities through fake emails or websites.

This project builds a **Phishing Email Detection Framework** using Natural Language Processing (NLP) and Machine Learning to automatically detect and classify phishing emails. The system is trained on real-world phishing and legitimate email data collected from multiple live sources.

As **Person 1 (Data Engineer)**, my core responsibility was to **collect, crawl, clean, label, and prepare the dataset** that the entire team uses for model training and evaluation.

---

## 👤 My Role — Data Engineer

As the Data Engineer of this project, I was responsible for building the foundation of the entire pipeline — the dataset. Without clean, labeled, and well-structured data, the NLP model cannot be trained or evaluated properly.

| Task                    | Description                                                                     |
| ----------------------- | ------------------------------------------------------------------------------- |
| **Data Collection**     | Collected phishing and legitimate email data from multiple real-world platforms |
| **API Integration**     | Integrated OpenPhish and PhishStats REST APIs for live phishing data            |
| **Web Crawling**        | Built a custom crash-safe web crawler to collect large scale phishing data      |
| **Data Labeling**       | Labeled all records properly — Phishing = 1, Legitimate = 0                     |
| **Data Cleaning**       | Removed nulls, duplicates, empty records, and inconsistent entries              |
| **Dataset Preparation** | Saved final structured dataset as CSV for Person 2 (NLP Preprocessing)          |

---

## 🌐 Data Collection

Three data sources were successfully used to build the final dataset:

---

### ✅ Source 1 — OpenPhish (Live Phishing URLs)

**Website:** https://openphish.com

OpenPhish is a fully automated phishing intelligence platform that continuously crawls the internet and identifies phishing URLs in real time. It maintains a live feed of currently active phishing URLs that is freely accessible without requiring any API key or registration.

**How It Works:**

> A simple HTTP GET request to the OpenPhish feed URL returns a plain text list of live phishing URLs — all verified and currently active at the time of collection. Each URL in the feed represents a real phishing threat that is active on the internet right now.

**Why It Is Useful:**

- No API key or registration required
- Returns **real-time** phishing URLs that are currently active
- Each URL represents a verified live phishing threat
- Simple and reliable — single HTTP request returns all data

```python
import requests

print("Fetching live phishing URLs from OpenPhish...")
response = requests.get("https://openphish.com/feed.txt")
phishing_urls = response.text.strip().split("\n")

print(f"Total phishing URLs fetched: {len(phishing_urls)}")

# Sample URLs fetched:
# https://username3586.invoice-ads-agency.com/
# https://www.roblox.com.ml/users/114667414927/profile
# https://www.confirm-address-check.vercel.app/
```

```
Status  : ✅ Success
Records : 300 live phishing URLs
Label   : 1 (Phishing)
Source  : openphish
```

---

### ✅ Source 2 — PhishStats REST API + Custom Web Crawler

**Website:** https://phishstats.info

PhishStats is a platform dedicated to fighting phishing and cybercrime since 2014. It gathers, enhances, and shares phishing intelligence with the cybersecurity community. It provides a REST API for accessing a large database of phishing records with detailed metadata including URL, IP address, country, score, and more.

**How It Works:**

> The PhishStats REST API (`https://api.phishstats.info/api/phishing`) was discovered through their official API documentation. The API supports filtering, sorting, and pagination. Since the API has a maximum limit of **100 records per request** and we needed 19,000 records, a custom web crawler was built to automatically paginate through 190 pages.

**API Details:**

- Base URL: `https://api.phishstats.info/api/phishing`
- Max records per request: 100
- Rate limit: 20 requests per minute
- Supports sorting, filtering, and pagination

```python
import requests
import time

for page in range(1, 191):
    url = f"https://api.phishstats.info/api/phishing?_sort=-date&_p={page}&_size=100"
    response = requests.get(url, timeout=10)
    data = response.json()
    # Extract URLs from each page
    time.sleep(3)  # Respect 20 requests/minute rate limit
```

**Sample API Response:**

```json
{
  "url": "https://login.zh-mobi-aiyouxisports.com/",
  "ip": "172.67.181.223",
  "score": 9.6,
  "date": "2026-04-26T00:55:05.000Z",
  "host": "zh-mobi-aiyouxisports.com",
  "tld": "com"
}
```

```
Status  : ✅ Success
Records : 19,000 phishing URLs (190 pages × 100 records)
Label   : 1 (Phishing)
Source  : phishstats_crawler
```

---

### ⏳ Source 3 — Enron Email Dataset (Legitimate Emails)

**Website:** https://www.cs.cmu.edu/~enron/

The Enron Email Dataset is one of the most widely used datasets in email research and NLP. It was collected during the investigation of the Enron Corporation scandal and contains over 500,000 real emails from approximately 150 Enron employees, mostly senior management. It is considered the gold standard for legitimate email data in cybersecurity research.

**Why Enron:**

- Real-world legitimate emails — not synthetic or generated
- Large volume — 500,000+ emails available
- Publicly available — hosted by Carnegie Mellon University (CMU)
- Widely used in academic NLP research — trusted and verified source
- CMU server confirmed accessible (HTTP Status 200 verified)

**Plan:**

> Download the dataset directly from CMU server, extract and parse individual email files, pull out subject and body text, label all records as legitimate (label = 0), and combine with our phishing data for the final dataset.

```python
url = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
# Download → Extract → Parse → Label 0 → Save
# Target: 20,000 legitimate emails
```

```
Status  : ⏳ In Progress
Records : Target 20,000 legitimate emails
Label   : 0 (Legitimate)
Source  : enron_crawler
```

---

## 🕷️ Web Crawler — Details

Since the PhishStats API limits responses to 100 records per request, a custom crash-safe web crawler was built to collect 19,000 records across 190 pages automatically.

### Crawler Features

| Feature                 | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| **Auto Save**           | Saves data to CSV every 10 pages — no data loss even if interrupted     |
| **Checkpoint System**   | Saves the last completed page number — allows full resume after crash   |
| **Rate Limit Handling** | Detects HTTP 429 errors and waits 60 seconds before retrying            |
| **Crash Recovery**      | Saves all collected data immediately when any unexpected error occurs   |
| **Progress Tracking**   | Prints live progress update every 10 pages                              |
| **Resume Support**      | On restart, automatically reads checkpoint and continues from last page |

### Crawler Progress Output

```
Crawling PhishStats — 19,000 records...
Estimated time: ~10 minutes
==================================================
Progress: Page 10/190  — Records: 1000  — Saved ✅
Progress: Page 20/190  — Records: 2000  — Saved ✅
Progress: Page 30/190  — Records: 3000  — Saved ✅
...
Progress: Page 190/190 — Records: 19000 — Saved ✅
Crawl Complete!
```

---

## 🧹 Data Cleaning

Raw data collected from APIs and crawlers is never perfect. It contains missing values, empty records, duplicate entries, and inconsistent formatting. The following cleaning steps were applied to ensure the dataset is ready for NLP processing:

| Step                 | Action                                               | Reason                                            |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------- |
| **Rename Columns**   | `Email Text` → `body`, `Email Type` → `label`        | Standardize column names across all sources       |
| **Label Encoding**   | `Phishing Email` → `1`, `Safe Email` → `0`           | Convert text labels to numeric for ML model       |
| **Drop Nulls**       | Remove rows where `body` is null                     | Empty emails have no text features for NLP        |
| **Drop Empty**       | Remove rows where `body` is blank or whitespace only | Whitespace-only text is useless for analysis      |
| **Drop Duplicates**  | Remove duplicate `body` entries                      | Duplicate data biases the model training          |
| **Strip Whitespace** | Remove leading and trailing spaces from all text     | Ensures clean text input for NLP preprocessing    |
| **Fix Labels**       | Keep only rows where label is exactly 0 or 1         | Remove any inconsistent or corrupted label values |
| **Reset Index**      | Reset DataFrame index after all cleaning steps       | Ensures clean sequential indexing for the model   |

### Final Column Structure

| Column    | Type | Description                     | Example                     |
| --------- | ---- | ------------------------------- | --------------------------- |
| `label`   | int  | 1 = Phishing, 0 = Legitimate    | `1`                         |
| `subject` | str  | Email subject line              | `"Verify your account now"` |
| `body`    | str  | Email body text or phishing URL | `"https://fake-login.com"`  |
| `source`  | str  | Where the data came from        | `"phishstats_crawler"`      |

---

## 🔄 Final Dataset Pipeline

```
OpenPhish ✅ ─────────────────────── 300   Phishing URLs  (label=1)
         ↓
PhishStats REST API + Crawler ✅ ─── 19,000 Phishing URLs  (label=1)
         ↓
Enron CMU Dataset ⏳ ─────────────── 20,000 Legitimate Emails (label=0)
         ↓
         ▼
┌──────────────────────────────────┐
│       COMBINE ALL SOURCES        │
│   Phishing (1) + Legitimate (0)  │
└──────────────────────────────────┘
         ↓
┌──────────────────────────────────┐
│           CLEAN DATA             │
│  Drop Nulls, Drop Duplicates,    │
│  Fix Labels, Strip Whitespace    │
└──────────────────────────────────┘
         ↓
┌──────────────────────────────────┐
│   phishing_dataset_final.csv     │
│  label | subject | body | source │
└──────────────────────────────────┘
         ↓
   Hand off to Person 2
   (NLP Preprocessing) ✅
```

---

## 📊 Dataset Summary

| Source              | Type              | Records     | Label     | Status         |
| ------------------- | ----------------- | ----------- | --------- | -------------- |
| OpenPhish Live Feed | Phishing URLs     | 300         | 1         | ✅ Done        |
| PhishStats Crawler  | Phishing URLs     | 19,000      | 1         | ✅ Done        |
| Enron CMU Dataset   | Legitimate Emails | 20,000      | 0         | ⏳ In Progress |
| **Total**           |                   | **~39,300** | **0 & 1** |                |

---

## 🛠️ Technologies Used

| Technology       | Purpose                                  |
| ---------------- | ---------------------------------------- |
| `Python`         | Core programming language                |
| `Pandas`         | Data manipulation and CSV handling       |
| `Requests`       | API calls and HTTP requests              |
| `BeautifulSoup4` | HTML parsing for web crawling            |
| `Matplotlib`     | Data visualization and graphs            |
| `Email`          | Parsing raw Enron email files            |
| `Tarfile`        | Extracting Enron .tar.gz archive         |
| `Time`           | Rate limit handling between API requests |
| `OS`             | File and checkpoint management           |

---

## 📁 File Structure

```
Person1/
│
├── 📓 data_collection.ipynb        ← Main notebook with all steps
├── 🕷️  crawler.py                  ← PhishStats web crawler script
├── 📊 phishstats_crawled.csv       ← Raw crawled data from PhishStats
├── 📊 enron_legitimate.csv         ← Legitimate emails from Enron (planned)
├── 📊 phishing_dataset_final.csv   ← Final combined and cleaned dataset
├── 🖼️  dataset_overview.png        ← Visualization graphs
├── 📄 checkpoint.txt               ← Crawler resume checkpoint file
└── 📄 README.md                    ← This file
```

---

## ⚙️ How to Run

### 1. Install Requirements

```bash
pip install pandas requests beautifulsoup4 matplotlib
```

### 2. Fetch OpenPhish Data

```python
import requests
response = requests.get("https://openphish.com/feed.txt")
phishing_urls = response.text.strip().split("\n")
print(f"Fetched: {len(phishing_urls)} phishing URLs")
```

### 3. Run PhishStats Crawler

```bash
python crawler.py
```

### 4. Check Crawler Progress Anytime

```python
import pandas as pd
df = pd.read_csv("phishstats_crawled.csv")
print(f"Records saved : {len(df)}")
print(f"Progress      : {len(df)/19000*100:.1f}%")
print(f"Remaining     : {19000 - len(df)}")
```

---

## ✅ Task Checklist

| Task                                           | Status                    |
| ---------------------------------------------- | ------------------------- |
| Collect phishing data from OpenPhish API       | ✅ Done — 300 records     |
| Collect phishing data via PhishStats crawler   | ✅ Done — 19,000 records  |
| Collect legitimate emails from Enron           | ⏳ In Progress            |
| Combine phishing + legitimate into one dataset | ⏳ After Enron collection |
| Label data properly — Phishing=1, Legitimate=0 | ✅ Done                   |
| Handle missing and inconsistent entries        | ✅ Done                   |
| Store dataset in structured CSV format         | ✅ Done                   |
| Visualize dataset distribution                 | ✅ Done                   |
| Hand off final dataset to Person 2             | ⏳ After Enron collection |

---

## 🤝 Team Collaboration

| Person                      | Role                       | Gets From Me                        |
| --------------------------- | -------------------------- | ----------------------------------- |
| **Person 1 — Harshit Ekka** | Data Engineer              | —                                   |
| **Person 2**                | NLP Preprocessing          | `phishing_dataset_final.csv`        |
| **Person 3**                | Model Developer            | Preprocessed features from Person 2 |
| **Person 4**                | Evaluation & Visualization | Trained model from Person 3         |

---

## 👨‍💻 Author

- **Name:** Harshit Ekka
- **Role:** Person 1 — Data Engineer
- **Project:** Phishing Email Detection Framework
