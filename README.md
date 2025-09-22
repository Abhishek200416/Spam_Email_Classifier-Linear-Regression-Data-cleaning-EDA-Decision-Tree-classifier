# Spam_Email_Classifier-Linear-Regression-Data-cleaning-EDA-Decision-Tree-classifier
Spam Email Classifier · Linear Regression · Titanic EDA · Decision Tree (Drug)

Task 1 — Three Coding Challenges (Machine Learning)

✅ Linear Regression on a housing dataset
✅ Data Cleaning & EDA on Titanic dataset
✅ Decision Tree classifier on Drug dataset

This repo contains three self-contained ML exercises implemented in Jupyter notebooks, plus a utility function SAFE_READ_CSV that makes running locally painless (no Google Colab dependence). Each challenge is written to be robust to dataset paths: just place the CSV next to the notebook or provide a full path when prompted.

🔧 What’s Inside
ML-low/
├─ Linear Regression-Data cleaning-EDA-Decision Tree classifier.ipynb   # Master notebook (all 3)
├─ Spam_Email_Classifier_Patched.ipynb                                  # (extra work)
├─ output.pdf                                                           # Exported report (example)
├─ README.md
└─ data/                                                                # (optional) put CSVs here

📦 Setup

Python 3.9+ recommended

Create/activate a venv and install deps:

python -m venv .venv
# Windows PowerShell
. .\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt


requirements.txt (use this if you don’t already have one):

pandas
numpy
matplotlib
scikit-learn
ipykernel
jupyter


If you plan to export to PDF:

pip install nbconvert[webpdf]


(You need a Chromium install; nbconvert will guide you.)

📁 Datasets (expected names)

Put these files next to the notebook or in ./data/:

Housing: Housing.csv (any standard housing dataset; the notebook will auto-detect target if named price or MEDV)

Titanic: Titanic-Dataset.csv (columns like Age, Embarked, Cabin, etc.)

Drug: drug200.csv (UCI-style sample with Drug as target)

You can also provide any full file path when prompted — thanks to SAFE_READ_CSV.

🛡️ Utility — SAFE_READ_CSV

To avoid brittle paths, all loaders call:

def SAFE_READ_CSV(preferred_paths, fallback_msg):
    # Tries multiple locations; if not found, prompts for a full path.


Tries known paths: ["/mnt/data/...","./...","data/..."]

If not found, prompts: paste a full path (e.g., C:\Users\...\data\Housing.csv)

Raises a clear error if still missing

🚀 How to Run

Open the master notebook and run all cells:

Linear Regression-Data cleaning-EDA-Decision Tree classifier.ipynb


Or run them section-by-section as you like.

Optional: export to PDF

jupyter nbconvert --to webpdf "Linear Regression-Data cleaning-EDA-Decision Tree classifier.ipynb"

📊 Challenge 1 — Linear Regression (Housing)

Goal: Predict house prices using LinearRegression.

Pipeline (already coded):

Load Housing.csv via SAFE_READ_CSV.

Target detection: if a column equals price or MEDV (case-insensitive), use it. Otherwise, the last numeric column becomes the target.

Features: all numeric columns except target.

Train/test split (test_size=0.2, random_state=42).

Fit LinearRegression.

Metrics: R², MAE.

Plot: first numeric feature vs. target (Actual vs. Predicted).

Outputs you’ll see:

Printed target column name

R² and MAE in console

Coefficients per feature

A scatter plot comparing actual vs. predicted along the first feature

💡 Tip: If your housing CSV includes non-numeric columns (like chas, rad codes as strings), convert them to numeric or drop them before training.

🧼 Challenge 2 — Data Cleaning & EDA (Titanic)

Goal: Clean Titanic-Dataset.csv and inspect structure.

Steps implemented:

df.info() and head() to understand schema

Missing values + duplicate counts (before/after)

Cleaning rules:

Age → fill with median

Embarked → fill with mode

Drop Cabin if present (often sparse)

Remove duplicates

Show missing values after cleaning

Preview cleaned frame

📌 You can extend with:

Survival rates by Sex, Pclass

Binning Age groups

Boxplots/histograms for quick visual EDA

🌳 Challenge 3 — Decision Tree Classifier (Drug)

Goal: Predict Drug using a shallow, interpretable decision tree.

Steps implemented:

Load drug200.csv via SAFE_READ_CSV

Label-encode categorical features (excluding target Drug)

Train/test split (0.2, random_state=42)

Train DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)

Evaluate accuracy

Visualize the tree with plot_tree (feature names & class names shown)

Outputs you’ll see:

Printed test accuracy (e.g., Accuracy: 0.95xx)

A rendered decision tree (depth-limited for readability)

🧪 Want more performance? Try GridSearchCV over max_depth, min_samples_split, min_samples_leaf.