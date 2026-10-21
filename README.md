# 📧 Email Spam Detection with Machine Learning — OIBSIP Data Science Task 4

> **Oasis Infobyte Internship Program (OIBSIP)**  
> **Track:** Data Science | **Task 4:** Email Spam Detection with Machine Learning  
> **Dataset:** [SMS Spam Collection (UCI ML Repository)](https://archive.ics.uci.edu/dataset/28/sms+spam+collection) — 5,574 messages (4827 ham / 747 spam)

---

## 🎯 Objective

Build a **Natural Language Processing (NLP) binary classifier** that distinguishes **spam** emails/messages from legitimate (**ham**) messages, demonstrating end-to-end skills from raw text cleaning to model evaluation and deployment readiness.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.10 |
| **Data** | `pandas`, `numpy` |
| **NLP** | `nltk` (stopwords, PorterStemmer), `re` |
| **ML** | `scikit-learn` (TF-IDF, MultinomialNB, LogisticRegression, LinearSVC) |
| **Visualisation** | `matplotlib`, `seaborn`, `wordcloud` |
| **Environment** | Jupyter Notebook |

---

## ✅ Feature Checklist — Completion Status

| Requirement | Status | Reference |
|-------------|--------|-----------|
| Data loading + class distribution check (spam vs ham counts & %) | ✅ | Section 2 & 3 |
| Text preprocessing pipeline: lowercase, punctuation removal, stopword removal, stemming | ✅ | Section 4 |
| Feature extraction using **TF-IDF Vectorizer** with explanation | ✅ | Section 5 |
| Train/test split (80/20, stratified) | ✅ | Section 6 |
| Train at least **2 classifiers** (Multinomial NB + Logistic Regression + Linear SVM) | ✅ | Section 7 |
| Evaluation: accuracy, precision, recall, F1-score, confusion matrix | ✅ | Section 8 |
| Discussion: Why is **recall** particularly important for spam detection? | ✅ | Section 9 |
| (Bonus) WordCloud visualisations for spam & ham | ✅ | Section 10 |
| Clean, commented Jupyter Notebook | ✅ | Throughout |

---

## 📂 Folder Structure (as per OIBSIP guidelines)

```
OIBSIP/
└── DataScience-Task4-EmailSpamDetection/
    ├── Email_Spam_Detection.ipynb   # Main notebook (executed, with outputs)
    ├── spam.csv                     # Dataset (TSV, 5,574 rows)
    ├── README.md                    # This file
    ├── requirements.txt             # Dependencies
    ├── metrics.json                 # Saved metrics
    ├── tfidf_vectorizer.pkl         # Trained TF-IDF vectorizer (optional)
    ├── best_spam_model.pkl          # Best model (Logistic Regression)
    └── screenshots/                 # Exported plots for LinkedIn/README
```

**GitHub push path (strict format):**
```
OIBSIP/DataScience-Task4-EmailSpamDetection/
```

---

## 📥 Dataset

- **Source:** UCI Machine Learning Repository — SMS Spam Collection (Tiago A. Almeida & José María Gómez Hidalgo, 2012)
- **Mirror:** `https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv`
- **Kaggle Alternative:** Search “SMS Spam Collection Dataset” on Kaggle
- **Shape:** 5,574 rows × 2 columns (`label`, `message`)
- **Distribution:** ham 86.6% (4,827) / spam 13.4% (747) — imbalanced
- **No download needed** — notebook tries local `spam.csv` then auto-downloads from GitHub raw if missing.

---

## 🔬 Methodology

### 1. EDA
- Null check, duplicate check, length/word-count stats
- Class distribution bar + pie charts
- Message length histograms & boxplots by class → spam is longer on average

### 2. Text Preprocessing Pipeline
```python
lowercase → remove punctuation/numbers → tokenise → remove stopwords → Porter stemming → rejoin
```
- Example:  
  `Free entry in 2 a wkly comp to win FA Cup...` → `free entri wkli comp win fa cup...`

### 3. Feature Extraction — TF-IDF
**TF-IDF = Term Frequency × Inverse Document Frequency**
- TF: local importance (how often a word appears in this message)
- IDF: `log(N / df)` — boosts rare discriminative words (`claim`, `prize`, `urgent`) and down-weights ubiquitous words (`you`, `go`)
- Config: `max_features=5000`, `ngram_range=(1,2)` (unigrams + bigrams like “free prize”), `min_df=2`, `max_df=0.95`

### 4. Train/Test Split
- 80/20, `stratify=y`, `random_state=42` → preserves 86/14 ratio in both splits
- **TF-IDF fitted only on train** to avoid leakage

### 5. Models Trained (3 for robustness)
| Model | Params | Rationale |
|-------|--------|-----------|
| Multinomial Naive Bayes | default | Classic for text, fast, strong baseline |
| Logistic Regression | `solver=liblinear`, `class_weight=balanced` | Handles sparse high-dim, interpretable |
| Linear SVM | `LinearSVC`, `class_weight=balanced` | Max-margin, often top accuracy on text |

### 6. Evaluation (Test set n=1,115)
| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|-------|----------|------------------|---------------|-----------|
| **Logistic Regression 🏆** | **0.9767** | 0.902 | **0.9262** | **0.9139** |
| Linear SVM | 0.9758 | 0.9122 | 0.9060 | 0.9091 |
| Multinomial NB | 0.9695 | 0.9832 | 0.7852 | 0.8731 |

> **Best Model: Logistic Regression** — highest F1 (0.9139) and Recall (0.9262), best balance for spam safety.

- Confusion matrices plotted for each model (TN, FP, FN, TP)
- Classification reports & Precision-Recall curves included

### 7. Why Recall Matters (Section 9)
- **False Negative (spam missed → inbox)** = phishing/malware risk → high cost
- **False Positive (ham → spam folder)** = inconvenience → lower cost
- Therefore **Recall = TP/(TP+FN)** (spam catch rate) is primary; Accuracy is misleading (86% by predicting all ham, but Recall=0)
- Our best Recall ~92.6% → only ~7% spam slips through; production adds blacklists & user feedback loops.

### 8. WordClouds & Error Analysis
- Ham cloud: conversational (“go”, “get”, “love”, “time”)
- Spam cloud: transactional (“free”, “claim”, “win”, “prize”, “txt”, “urgent”)
- 5 misclassified examples inspected: FN often lack classic keywords or use obfuscation; FP often ham with promotional language.

---

## 🚀 How to Run

```bash
# 1. Clone your OIBSIP repo
git clone https://github.com/<your-username>/OIBSIP.git
cd OIBSIP/DataScience-Task4-EmailSpamDetection

# 2. Install deps
pip install -r requirements.txt

# 3. (Optional) Download NLTK data — notebook does this automatically
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 4. Launch notebook
jupyter notebook Email_Spam_Detection.ipynb
# Or run headless execution:
jupyter nbconvert --to notebook --execute Email_Spam_Detection.ipynb --output executed.ipynb
```

> **No API keys needed.** Dataset is bundled as `spam.csv`. If deleted, notebook auto-fetches from GitHub raw.

---

## 📊 Key Visuals (Embedded in Notebook)

- Class distribution (bar + pie)
- Message length histograms & word-count boxplots
- Model comparison bar chart (Accuracy/Precision/Recall/F1)
- 3× confusion-matrix heatmaps with TN/FP/FN/TP annotations
- Precision-Recall curves
- WordClouds (ham vs spam) + Top-20 word bar charts

All figures are rendered inline in the executed notebook. Exported PNGs are in `screenshots/` for LinkedIn posts.

---

## 🔮 Custom Prediction Demo

The notebook ends with a live demo on unseen custom messages:

```python
"Congratulations! You have won a $1000 cash prize..."  → SPAM 🚨
"Hey, are we still on for dinner tonight?"              → HAM ✅
"URGENT: Your account has been compromised..."          → SPAM 🚨
"Happy birthday! Hope you have a wonderful day :)"      → HAM ✅
```

---

## 📈 Limitations & Next Steps

| Current | Production Upgrade |
|---------|-------------------|
| Bag-of-words TF-IDF | Transformer embeddings (BERT/DistilBERT) |
| Static 5k vocab | Subword tokenisation, multilingual |
| No metadata | Sender reputation, SPF/DKIM, URL blacklists |
| No online learning | User “Report spam” feedback loop |

**Improvements:** character n-grams for obfuscation (`fr33`), URL/phone regex features, XGBoost/LightGBM, Flask API.

---

## 📚 References

1. UCI SMS Spam Collection — Almeida & Gómez Hidalgo (2012)
2. scikit-learn TF-IDF docs — https://scikit-learn.org/stable/modules/feature_extraction.html
3. NLTK — https://www.nltk.org/
4. Kaggle SMS Spam Collection — https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
5. Manning et al. — *Introduction to Information Retrieval* (TF-IDF theory)

---

## 🎥 Demo Video Guidelines (OIBSIP Requirements)

Per Section 1.2 Step 4:
- First **2 seconds**: static title card with **Full Name + Assigned Track (Data Science) + Task Title (Email Spam Detection)**
- Then: screen-recorded walkthrough of notebook running end-to-end (not just screenshots)
- Show: EDA plots, preprocessing demo, TF-IDF explanation, model comparison, confusion matrices, WordClouds, custom predictions
- Upload to LinkedIn, tag **Oasis Infobyte**, add **#oasisinfobyte** + **#datascience** + **#machinelearning**

---

## 📤 Submission Checklist (OIBSIP)

- [x] Notebook executed with all outputs visible
- [x] Code commented, markdown explanations between charts
- [x] Pushed to `OIBSIP/DataScience-Task4-EmailSpamDetection/` on GitHub (repo name exactly **OIBSIP**)
- [ ] Recorded demo video (2-sec title card → end-to-end demo)
- [ ] LinkedIn post with video, tagged Oasis Infobyte, hashtag #oasisinfobyte
- [ ] Substantive comments on ≥2 peer videos
- [ ] Submitted via Task Submission Form (link in welcome email) with GitHub repo URL
- [ ] Await evaluation (~2 weeks) → Completion Certificate + LOR (if eligible)

---

## 📧 Contact & Support

- **Email:** support@oasisinfobyte.com
- **Telegram:** https://t.me/oasisinfobyte
- **LinkedIn:** https://in.linkedin.com/company/oasis-infobyte
- **Website:** https://www.oasisinfobyte.com

---

*“Learning is the only thing the mind never exhausts, never fears, and never regrets.” — Leonardo da Vinci*

**Made with ❤️ for OIBSIP Data Science Internship**
