# ✉️ Message Intelligence System

**Spam vs Legitimate message classification using KNN, SVM and Naive Bayes**


| Deliverable | Link |
|---|---|
| 🚀 Live Streamlit App | [messageintelligencesystemsl.streamlit.app](https://messageintelligencesystemsl.streamlit.app/) |
| 🎥 Video Explanation | [Watch on Google Drive](https://drive.google.com/file/d/109jQe46a_nf6DF0fp1fADjR6mhRsRRAL/view?usp=sharing) |


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
- **Missing values:** `sender_activity_score` (106), `sender_account_age_days` (113) and `messages_sent_last_24h` (162), filled with the median

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

The notebook follows the project brief part by part:

1. **Part A, Theory:** conditional probability, Bayes' theorem, Naive Bayes assumptions, working principles of KNN and SVM, and a comparison of distance-based, probabilistic and margin-based classifiers.
2. **Part B, Preparation:** features and target identified, median imputation, standard scaling and an 80/20 train-test split (`random_state=50`).
3. **Part C, KNN:** K from 1 to 19, three distance metrics (Euclidean, Manhattan, Minkowski) and a misclassification check.
4. **Part D, SVM:** linear, polynomial and RBF kernels, support-vector analysis and comparison with KNN.
5. **Part E, Naive Bayes:** Gaussian Naive Bayes, conditional probabilities for sample messages, a Bayes' theorem calculation for `P(Spam | HasURL)` and a comparison with the model output.
6. **Part F, Evaluation:** accuracy, precision, recall, F1, confusion-matrix counts and 5-fold stratified cross-validation.
7. **Part G, Report:** strengths and weaknesses, impact of Naive Bayes assumptions, interpretability trade-offs and a business recommendation.

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

- **KNN:** scores 1.0 for K = 1 to 15 with every distance metric tested. At K = 17 and 19, recall drops to 0.9947 (accuracy 0.9990) because a larger K lets legitimate neighbours outvote a few spam messages.
- **SVM:** the linear SVM uses only 13 of 4,160 training points (0.31%) as support vectors.
- **Naive Bayes:** the hand-calculated posterior `P(Spam | HasURL) = 0.6695` matches the model output exactly, so Naive Bayes is Bayes' theorem plus the independence assumption.
- **Best for precision and recall:** all five models tie at 1.0 on the test set, so accuracy cannot rank them. KNN is the only model that loses recall (at large K and in cross-validation).
- **Why the scores are perfect:** the classes are almost perfectly separable (no legitimate message contains a special character, while every spam message does), so a single feature separates them.

### Charts

| | |
|---|---|
| ![Correlation matrix](Screenshots/correlation_matrix.png) | ![Model comparison](Screenshots/model_comparison.png) |

---

## 🏆 Business Recommendation

- **Primary model: Linear SVC.** It matches every other model, predicts quickly, needs no kernel tuning, and its feature weights explain each decision. Its signed distance from the decision boundary is used as the spam score.
- **Naive Bayes** is useful for explanation and cross-checking, but not for reporting probabilities, because they are overconfident.
- Set the decision threshold by business cost and send borderline messages to **human review**.
- This dataset is probably easier than real traffic, so validate on fresh labelled messages before launch and retrain as spam patterns change.

---

## 📁 Repository Structure

```
├── app.py                  # Streamlit app
├── train_model.py          # Trains and exports the Linear SVC pipeline
├── requirements.txt
├── README.md
├── Dataset/
│   └── Message_Intelligence_Dataset_5200.xlsx
├── model/
│   ├── spam_svc_pipeline.pkl
│   └── metrics.json
├── Notebook/
│   └── Message_Intelligence_System.ipynb
└── Screenshots/
```

---

## ⚙️ How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train and export the model
python train_model.py

# 3. Start the app
streamlit run app.py
```

`train_model.py` uses the same split as the notebook (`test_size=0.2`, `random_state=50`) and writes `model/spam_svc_pipeline.pkl` and `model/metrics.json`.

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Matplotlib · scikit-learn · Streamlit · Jupyter Notebook

---

## 👤 Author

**Harshal Vora**
GitHub: [HarshalVora86](https://github.com/HarshalVora86) · LinkedIn: [Harshal Vora](https://www.linkedin.com/in/harshal-vora-344601251/)
