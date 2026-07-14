# End-to-End Data Preprocessing Pipeline

This repository contains a complete, automated data preprocessing and feature engineering pipeline implemented in a unified Python script (`main.py`). The pipeline processes an insurance dataset to clean, analyze, visualize, and structurally transform features into a machine-learning-ready framework.

## 📁 Repository Structure

```text
AIML_Assignment_18/
│
├── insurance_data.csv          # Source dataset containing demographics and charges
├── main.py                     # Unified script executing tasks Q1 to Q10
└── README.md                   # Project documentation and guide
```

---

## 🛠️ Step-by-Step Implementation

### Phase 1: Data Profiling & Cleaning
* **Q1. Data Loading & Structural Verification:** Automatically reads inputs, prints structural matrix boundaries (`.shape`), and evaluates raw underlying structural classifications (`.dtypes`).
* **Q2. Missing & Duplicate Filtration:** Scans for null coordinates and drops duplicate records using `.drop_duplicates()` to safeguard statistical models from valuation bias.
* **Q3. Mathematical Summary Summary:** Generates continuous metric breakdowns using descriptive statistical properties (minimum, maximum, average median thresholds).

### Phase 2: Exploratory Data Analysis (EDA)
* **Q4. Distribution Histograms (Graph 1):** Graphs data density layout profiles across structural features (`age`, `bmi`, `children`, `charges`) using integrated Kernel Density Estimate curves.
* **Q5. Categorical Category Counts (Graph 2):** Renders item population proportions across target categorical segments (`sex`, `smoker`, `region`).
* **Q6. Interaction Heatmaps (Graph 3):** Combines metric attributes into a visual cross-correlation color grid using custom colormaps to evaluate target dependency strengths.

### Phase 3: Machine Learning Feature Engineering
* **Q7. Structural Partition Split:** Segregates independent demographic input features ($X$) away from target dependent output expense parameters ($y$).
* **Q8. Vector One-Hot Encoding:** Passes categorical dimensions through Scikit-Learn's `OneHotEncoder(drop='first')` to generate binary input columns without falling into dummy-variable correlation traps.
* **Q9. Scale Transformation Alignment:** Converts metric measure ranges using `StandardScaler` to realign continuous factors onto unit variance levels.
* **Q10. Core Composite Pipeline Assembly:** Unifies individual step pipelines directly inside a central `ColumnTransformer` object to safely eliminate algorithmic data leakage risks.

---

## 🚀 Execution Instructions

### Prerequisites
Ensure your local environment includes standard statistical modeling libraries. Install them via terminal console setup:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Project
1. Open your terminal window inside your project root folder (`AIML_Assignment_18`).
2. Launch the consolidated execution wrapper script:
   ```bash
   python main.py
   ```
3. **Important Interaction Detail:** The pipeline processes data linearly. Visual validation windows (Graphs 4, 5, and 6) will appear in sequence. Simply **Close (X)** each open graph configuration dialog box to trigger execution steps for trailing questions.
4.git add .
git commit -m "Added complete preprocessing code and README"
git branch -M main
git push -u origin main
