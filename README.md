# Spam_Email_Classifier-Linear-Regression-Data-cleaning-EDA-Decision-Tree-classifier# ML-Low — Task 1 & Task 2 (Jupyter Notebooks)

Practical, no-nonsense ML mini-projects you can run locally in VS Code/Jupyter.  
Covers **Linear Regression (Housing)**, **Titanic Data Cleaning & EDA**, **Decision Tree (Drug)**, and a **Spam Email Classifier (Naive Bayes)**.

---

## 📁 Repository Contents

ML-low/
├─ Linear Regression-Data cleaning-EDA-Decision Tree classifier.ipynb # Task 1: all 3 challenges in one notebook
├─ Spam_Email_Classifier_Patched.ipynb # Task 2: spam vs ham classifier
├─ output.pdf # Exported report (optional)
├─ README.md
└─ data/ # (create this; place CSVs here)
├─ Housing.csv
├─ Titanic-Dataset.csv
└─ drug200.csv

markdown
Copy code

> You can keep datasets next to the notebooks **or** in `data/`.  
> Both notebooks include a `SAFE_READ_CSV` utility that tries multiple default paths and, if needed, asks you for a manual path.

---

## 🧪 Task 1 — Three Coding Challenges (Single Notebook)

### 1) Linear Regression — Housing
**Goal:** Predict house prices using `LinearRegression`.

- **Dataset:** `Housing.csv`  
- **Pipeline:** load → select numeric features → train/test split → fit → evaluate (R², MAE) → quick scatter plot  
- **Auto-target logic:** If the dataset contains a `price/medv` column (case-insensitive), that’s used; otherwise the **last numeric column** becomes the target.

**Outputs:**
- R², MAE
- Coefficients per feature
- Scatter: first numeric feature vs. actual/predicted target

---

### 2) Data Cleaning & EDA — Titanic
**Goal:** Clean and inspect Titanic passenger data with Pandas.

- **Dataset:** `Titanic-Dataset.csv`  
- **Steps:** `.info()`, `.head()`, missing values, duplicates →  
  - Impute `Age` with median  
  - Impute `Embarked` with mode  
  - Drop `Cabin` if present  
  - Remove duplicates  
- **Outputs:** Missing-value tables (before/after) + cleaned preview

---

### 3) Decision Tree Classifier — Drug
**Goal:** Predict prescribed drug class using a small clinical dataset.

- **Dataset:** `drug200.csv`  
- **Steps:** label-encode categoricals (except target `Drug`) → train/test split → `DecisionTreeClassifier(criterion='entropy', max_depth=4)` → accuracy → visualize tree  
- **Outputs:** Test accuracy, plotted decision tree

---

## ✉️ Task 2 — Spam Email Classifier (Naive Bayes)

**Goal:** Train a spam/ham classifier using **pre-vectorized** Kaggle dataset (word counts) and **NB** with **TF-IDF weighting**.

- **Dataset (example):** Kaggle — *Email Spam Classification Dataset CSV*  
  - URL: https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv  
  - The CSV has ~3000 count features (+ a label column).  
- **Vectorization:** Since features are counts already, we use `TfidfTransformer` (not `TfidfVectorizer`).
- **Models:** `MultinomialNB` and `ComplementNB` with `GridSearchCV` on `alpha`.
- **Metrics:** accuracy, precision, recall, F1, confusion matrix; auto-selects best model.

> If your dataset has only two columns (`text`, `label`), you can adapt the notebook to use `TfidfVectorizer` on raw text—this notebook assumes the Kaggle **count-feature** CSV.

---

## 🧰 SAFE_READ_CSV Utility

Both notebooks define:

```python
def SAFE_READ_CSV(preferred_paths, fallback_msg):
    # Tries a list of locations.
    # If not found, prompts for a manual path (works outside Colab).
Typical usage in the notebooks tries:

local folder: filename.csv

project subfolder: data/filename.csv

(and a Linux-ish path for hosted runners)

If not found, you’ll get a prompt:

pgsql
Copy code
➡ Enter full path to your CSV (or press Enter to cancel):
▶️ How to Run (VS Code / Jupyter)
Create env & install deps

bash
Copy code
# Python 3.10+ recommended
python -m venv .venv
.venv\Scripts\activate         # Windows PowerShell
pip install -U pip
pip install jupyter pandas numpy scipy scikit-learn matplotlib
Open notebook

VS Code → Jupyter extension → open .ipynb and select your .venv kernel, or

bash
Copy code
jupyter notebook
# then open the .ipynb in the browser
Place datasets

Put CSVs in data/ (recommended) or next to the notebook.

If the loader can’t find them, paste the full path when prompted.

📝 Expected Files & Where to Put Them
data/Housing.csv

data/Titanic-Dataset.csv

data/drug200.csv

(For Task 2) A Kaggle emails CSV (count-features): place as data/email_spam.csv (or provide path when prompted)

✅ Deliverables Check
Task 1 (single notebook):

Prints metrics for Linear Regression (R², MAE)

Shows Titanic cleaning before/after summaries

Prints Decision Tree accuracy + renders tree

Task 2:

Loads Kaggle CSV

Runs TF-IDF transform on count features

Tunes alpha for NB models via GridSearchCV

Prints best model + full classification report & confusion matrix

🧪 Want more performance? Try GridSearchCV over max_depth, min_samples_split, min_samples_leaf.
