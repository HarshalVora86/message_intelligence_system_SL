# ✉️ Message Intelligence System

**Spam vs Legitimate message classification using KNN, SVM and Naive Bayes**

Supervised Learning, Project 4 | Author: Harshal Vora ([HarshalVora86](https://github.com/HarshalVora86))

| Deliverable | Link |
|---|---|
| 🚀 Live Streamlit App | `PASTE_STREAMLIT_URL_HERE` |
| 🎥 Video Explanation | `PASTE_GOOGLE_DRIVE_OR_YOUTUBE_UNLISTED_LINK_HERE` |
| 📓 Notebook | [`Notebook/Message_Intelligence_System.ipynb`](Notebook/Message_Intelligence_System.ipynb) |
|

---

## 📌 Project Overview

A communication security company wants to classify incoming messages automatically as:

- `0` → Legitimate message
- `1` → Spam message

This project builds and compares three families of classifiers and explains their behaviour with probability concepts (conditional probability and Bayes' theorem):

| Family | Model |
|---|---|
| Distance-based | K-Nearest Neighbors (KNN) |
| Margin-based | Support Vector Machine (Linear, Polynomial, RBF) |
| Probabilistic | Gaussian Naive Bayes |

---

## 📂 Dataset

`Dataset/Message_Intelligence_Dataset_5200.xlsx`: 5,200 messages, 16 columns, no duplicates.

- **Class balance:** 4,227 legitimate (81.3%) and 973 spam (18.7%)
- **Missing values:** `sender_activity_score` (106), `sender_account_age_days` (113), `messages_sent_last_24h` (162), filled with the median

| Feature | Description |
|---|---|
| `message_length`, `word_count` | Size of the message |
| `num_urls`, `num_digits`, `num_special_chars` | Content signals |
| `spam_keyword_score`, `legit_keyword_score` | Keyword frequency scores |
| `sender_activity_score`, `sender_account_age_days`, `messages_sent_last_24h` | Sender behaviour |
| `hour_of_day`, `day_of_week` | Time features |
| `spam_label` | **Target** (0 = Legitimate, 1 = Spam) |

`message_id`, `message_text` and `timestamp` are dropped before modelling.

---

## 🔬 Methodology

1. **Theory (Part A):** conditional probability, Bayes' theorem, Naive Bayes assumptions, KNN and SVM working principles, distance vs probabilistic vs margin-based classifiers.
2. **Preparation (Part B):** features and target identified, median imputation, standard scaling, 80/20 train-test split (`random_state=50`).
3. **KNN (Part C):** K from 1 to 19, three distance metrics (Euclidean, Manhattan, Minkowski), misclassification analysis.
4. **SVM (Part D):** linear, polynomial and RBF kernels, support-vector analysis, comparison with KNN.
5. **Naive Bayes (Part E):** Gaussian Naive Bayes, manual Bayes calculation for sample messages, comparison of theory with model output.
6. **Evaluation (Part F):** accuracy, precision, recall, F1, confusion matrices and 5-fold stratified cross-validation.
7. **Report (Part G):** strengths and weaknesses, impact of Naive Bayes assumptions, interpretability trade-offs, business recommendation.

---

## 📊 Results

### Test set (1,040 messages: 852 legitimate, 188 spam)

| Model | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| KNN (K = 5) | 1.0 | 1.0 | 1.0 | 1.0 | 852 | 0 | 0 | 188 |
| Linear SVC | 1.0 | 1.0 | 1.0 | 1.0 | 852 | 0 | 0 | 188 |
| Polynomial SVC | 1.0 | 1.0 | 1.0 | 1.0 | 852 | 0 | 0 | 188 |
| RBF SVC | 1.0 | 1.0 | 1.0 | 1.0 | 852 | 0 | 0 | 188 |
| Gaussian Naive Bayes | 1.0 | 1.0 | 1.0 | 1.0 | 852 | 0 | 0 | 188 |

### 5-fold stratified cross-validation (F1)

| Model | Mean F1 | Std |
|---|---|---|
| KNN | 0.9995 | 0.001 |
| Linear SVC | 1.0000 | 0.000 |
| Polynomial SVC | 1.0000 | 0.000 |
| RBF SVC | 1.0000 | 0.000 |
| Naive Bayes | 1.0000 | 0.000 |

### Key findings

- **KNN** is perfect for K = 1 to 15 with every distance metric tested. At K = 17 and 19, recall drops to 0.9947 because a larger K lets legitimate neighbours outvote a few spam messages.
- **SVM:** the linear SVM uses only 13 of 4,160 training points (0.31%) as support vectors.
- **Naive Bayes:** the hand-calculated posterior P(Spam | HasURL) = 0.6695 matches the model exactly, so Naive Bayes is Bayes' theorem plus the independence assumption.
- **Why the scores are perfect:** the classes are almost perfectly separable (no legitimate message contains a special character, every spam message does), so a single feature separates them and accuracy cannot rank the models.

---

## 🏆 Business Recommendation

- **Primary model: Linear SVC.** It matches every other model, predicts quickly, needs no kernel tuning, and its feature weights explain each decision. Its signed distance from the boundary is used as the spam score.
- **Naive Bayes** is useful for explanation and cross-checking, but not for reporting probabilities (they are overconfident).
- Set the decision threshold by business cost and send borderline messages to **human review**.
- This dataset is probably easier than real traffic, so validate on fresh labelled messages before launch and retrain as spam patterns change.

---

## 🚀 Streamlit App

The app loads the saved Linear SVC pipeline (median imputer + StandardScaler + model), so predictions use exactly the same preprocessing as training.

- Shows the **spam score** (signed distance from the decision boundary). A positive score is on the spam side.
- Messages inside the SVM margin (|score| < 1) are flagged as **borderline, send to human review**.
- A threshold slider lets the user trade off precision against recall.

```
User Input → DataFrame → Saved Pipeline → Median Imputer + StandardScaler
→ Linear SVC → Spam Score → Spam / Legitimate / Review
```

### Screenshots

| | |
|---|---|
| ![Legitimate message](Screenshots/app_legitimate.png) | ![Spam message](Screenshots/app_spam.png) |
| ![Model comparison](Screenshots/model_comparison.png) | ![KNN K sweep](Screenshots/knn_k.png) |

---

## 📁 Repository Structure

```
├── app.py                  # Streamlit app
├── train_model.py          # Trains and exports the pipelines
├── requirements.txt
├── README.md
├── Dataset/
│   └── Message_Intelligence_Dataset_5200.xlsx
├── model/
│   ├── spam_svc_pipeline.pkl
│   └── metrics.json
├── Notebook/
│   └── Message_Intelligence_System.ipynb
├── Screenshots/
└── Theory/
    └── Theory_Concepts.pdf
```

---

## ⚙️ How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train and export the models
python train_model.py

# 3. Start the app
streamlit run app.py
```

`train_model.py` uses the same split as the notebook (`test_size=0.2`, `random_state=50`) and writes `model/spam_svc_pipeline.pkl` and `model/metrics.json`.

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Matplotlib · scikit-learn · Streamlit · Jupyter Notebook

---

## ⚠️ Limitations

- 100% test scores and one perfectly separating feature mean this dataset is easier than real traffic.
- The app takes numeric message features as input, not raw message text.
- Spammers can adapt (for example by removing special characters or URLs), so the model needs monitoring and retraining.

---

## 👤 Author

**Harshal Vora**
GitHub: [HarshalVora86](https://github.com/HarshalVora86) · LinkedIn: [Harshal Vora](https://www.linkedin.com/in/harshal-vora-344601251/)
