# 🫀 Cardiac Risk Prediction — Decision Tree Classifier

> A machine learning pipeline for predicting cardiac risk in hospital patients.  
> Covers the full data science workflow: messy real-world data cleaning, feature engineering, and a Decision Tree classifier achieving ~89% accuracy.

![Python](https://img.shields.io/badge/Language-Python-blue)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![pandas](https://img.shields.io/badge/Data-pandas-150458)
![Status](https://img.shields.io/badge/Accuracy-89.47%25-brightgreen)

---

## ✨ Pipeline Overview

```
Raw CSV  →  Data Cleaning  →  Feature Engineering  →  Model Training  →  Evaluation
```

| Stage | Description |
|---|---|
| 🧹 Cleaning | Normalize inconsistent values (mixed Persian/English/boolean formats) |
| 🔢 Encoding | One-hot encode categorical columns; ordinal-encode severity scales |
| 📅 Date handling | Parse mixed-format dates; compute follow-up period in days |
| 🌲 Modeling | Decision Tree with depth limit to prevent overfitting |
| 📊 Evaluation | Accuracy, classification report, confusion matrix, feature importance |

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `Ds_Fater (2).csv` |
| Rows | 20,040 patients |
| Columns | 26 features |
| Target | `Target` — `0` Low Risk / `1` High Risk |
| Source hospitals | Imam, Shariati, Milad, Modarres |

### Key features

`Age` · `Sex` · `BMI` · `Troponin_Level` · `RestingBP` · `Cholesterol` · `MaxHR` · `CAD` · `Heart_Failure` · `Arrhythmia` · `Valvular_Disease` · `CHD` · `Diabetes` · `Smoking` · `Alcoholism` · `Depression_Stress` · `Department` · `Education_Level` · `CPT`

---

## 🧹 Data Cleaning Highlights

The raw dataset contained significant inconsistency across multiple columns:

**Mixed-format values normalized:**
- `Sex` → `1/0/M/F/man/woman` all mapped to `1` or `0`
- `Heart_Failure`, `Diabetes`, `Smoking`, `CHD` → mixed booleans, Persian (`بله/خیر`), and string variants all mapped to `1` or `0`
- `Valvular_Disease` → Persian severity scale (`بدون/خفیف/متوسط/شدید`) mapped to `0–3`

**Missing values handled:**

| Column | Strategy |
|---|---|
| `BMI` | Recalculated from `Weight` / `(Height/100)²` |
| `Diabetes`, `Smoking` | Filled with column mode |
| `Arrhythmia` | Filled with `'none'` |
| `Education_Level` | Filled with `'Unknown'` |
| `Last_Visit_Date` | Missing follow-up period set to `0` |

**Outlier capping:**
- `Alcoholism` capped at `50`
- `MaxHR` capped at `200`

---

## ⚙️ Model

```python
DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=2
)
```

| Metric | Value |
|---|---|
| Train accuracy | 89.55% |
| Test accuracy | 89.47% |
| Train / test split | 80% / 20% |

### Classification report (test set)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Low Risk (0) | 0.85 | 0.99 | 0.91 | 2,265 |
| High Risk (1) | 0.99 | 0.77 | 0.86 | 1,743 |
| **Weighted avg** | **0.91** | **0.89** | **0.89** | **4,008** |

> The model is highly precise at identifying High Risk patients (99%) but misses ~23% of them (lower recall). This trade-off may need tuning depending on clinical requirements.

---

## 📈 Visualizations

The notebook produces a 2×2 figure containing:

1. **Decision Tree diagram** — full tree visualization up to depth 4
2. **Top 10 feature importances** — horizontal bar chart
3. **Confusion matrix** — predicted vs actual risk labels
4. **Troponin Level by risk group** — box plot showing distribution difference

---

## 📁 Project Structure

```
cardiac-risk/
└── sep.ipynb       # Full pipeline: cleaning → model → evaluation
    Ds_Fater (2).csv     # Raw patient dataset (required)
```

---

## 🚀 Getting Started

### Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run

Open `sep.ipynb` in Jupyter and run all cells in order.

---

## 👩‍💻 Author

**Sepideh Pashayan** 
