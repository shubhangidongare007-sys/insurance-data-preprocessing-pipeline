import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =====================================================================
# TASK 1 / Q1. Data Loading & Initial Analysis
# =====================================================================
df = pd.read_csv('insurance_data.csv') 

print("==================== Q1: INITIAL DATA ANALYSIS ====================")
print("--- First 10 Rows ---")
print(df.head(10))

print("\n--- Last 5 Rows ---")
print(df.tail(5))

print(f"\nDataset Shape (Rows, Columns): {df.shape}")
print("\n--- Data Types of All Columns ---")
print(df.dtypes)


# =====================================================================
# TASK 2 / Q2. Missing & Duplicate Values
# =====================================================================
print("\n==================== Q2: MISSING & DUPLICATE VALUES ====================")
print("--- Missing Values per Column ---")
print(df.isnull().sum())

duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows found: {duplicate_count}")

df_cleaned = df.drop_duplicates(keep='first')
print(f"Dataset shape after removing duplicates: {df_cleaned.shape}")


# =====================================================================
# TASK 3 / Q3. Statistical Summary
# =====================================================================
print("\n==================== Q3: STATISTICAL SUMMARY ====================")
print("--- Statistical Summary of Numeric Columns ---")
print(df_cleaned.describe())

key_columns = ['age', 'bmi', 'charges']
print("\n--- Key Metrics For Selected Columns ---")
for col in key_columns:
    if col in df_cleaned.columns:
        print(f"\nColumn: {col}")
        print(f"  Minimum: {df_cleaned[col].min()}")
        print(f"  Maximum: {df_cleaned[col].max()}")
        print(f"  Mean:    {df_cleaned[col].mean():.2f}")
        print(f"  Median:  {df_cleaned[col].median()}")


# =====================================================================
# TASK 4 / Q4. Histogram of Numeric Features (GRAPH 1)
# =====================================================================
print("\n==================== Q4: GENERATING HISTOGRAMS ====================")
numeric_cols = ['age', 'bmi', 'children', 'charges']
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(data=df_cleaned, x=col, kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Distribution of {col}', fontsize=12)

plt.tight_layout()
print("-> Displaying Histogram Window. Close the graph window to continue to next task...")
plt.show() 


# =====================================================================
# TASK 5 / Q5. Count Plots of Categorical Features (GRAPH 2)
# =====================================================================
print("\n==================== Q5: GENERATING COUNT PLOTS ====================")
categorical_cols = ['sex', 'smoker', 'region']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, col in enumerate(categorical_cols):
    sns.countplot(
        data=df_cleaned, 
        x=col, 
        ax=axes[i], 
        palette='Set2', 
        order=df_cleaned[col].value_counts().index
    )
    axes[i].set_title(f'Count Plot of {col}', fontsize=12)

plt.tight_layout()
print("-> Displaying Count Plot Window. Close the graph window to continue to next task...")
plt.show() 


# =====================================================================
# TASK 6 / Q6. Correlation Heatmap (GRAPH 3)
# =====================================================================
print("\n==================== Q6: GENERATING CORRELATION HEATMAP ====================")
plt.figure(figsize=(8, 6))
numeric_df = df_cleaned.select_dtypes(include=[np.number])

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numeric Features', fontsize=14)
print("-> Displaying Heatmap Window. Close the graph window to continue to next task...")
plt.show() 


# =====================================================================
# TASK 7 / Q7. Feature Identification
# =====================================================================
print("\n==================== Q7: FEATURE IDENTIFICATION ====================")
X = df_cleaned.drop(columns=['charges'])
y = df_cleaned['charges']

print("Features successfully identified and separated.")
print(f"Independent Matrix Shape (X): {X.shape}")
print(f"Dependent Target Vector Shape (y): {y.shape}")


# =====================================================================
# TASK 8 / Q8. Encoding Categorical Variables
# =====================================================================
print("\n==================== Q8: ENCODING CATEGORICAL VARIABLES ====================")
cat_cols_to_encode = ['sex', 'smoker', 'region']

encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = encoder.fit_transform(X[cat_cols_to_encode])

print("--- Transformed Category Features Sample (First 5 Rows) ---")
print(encoded_features[:5])


# =====================================================================
# TASK 9 / Q9. Feature Scaling
# =====================================================================
print("\n==================== Q9: FEATURE SCALING ====================")
num_cols_to_scale = ['age', 'bmi', 'children']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X[num_cols_to_scale])

print("--- Scaled Numeric Features Sample (First 5 Rows) ---")
print(scaled_features[:5])


# =====================================================================
# TASK 10 / Q10. Mini Project - Complete Preprocessing Pipeline
# =====================================================================
print("\n==================== Q10: COMPLETE PREPROCESSING PIPELINE ====================")
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='first', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols_to_scale),
        ('cat', categorical_transformer, cat_cols_to_encode)
    ]
)

X_final_processed = preprocessor.fit_transform(X)

print(f"Complete Preprocessing Finalized. Cleaned Output Matrix Shape: {X_final_processed.shape}")
print("Functional machine learning pre-processing blueprint ready successfully without errors!")