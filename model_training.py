import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load the dataset ---
try:
    data = pd.read_csv('chronic_disease_dataset.csv')
except FileNotFoundError:
    print("Error: 'chronic_disease_dataset.csv' not found. Please place the dataset in the same directory.")
    exit()

# --- Data Preprocessing ---
X = data.drop('target', axis=1)
y = data['target']

categorical_features = ['gender', 'smoking_status', 'alcohol_intake', 'family_history']
numerical_features = X.columns.drop(categorical_features).tolist()

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# --- Model Definitions ---
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# --- Model Training and Evaluation ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

results = {}
best_model = None
best_accuracy = 0.0

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
    except TypeError as e:
        # Handle sparse matrix error for models that need dense data
        if 'sparse matrix' in str(e):
            X_train_dense = preprocessor.fit_transform(X_train)
            X_test_dense = preprocessor.transform(X_test)

            if sparse.issparse(X_train_dense):
                X_train_dense = X_train_dense.toarray()
                X_test_dense = X_test_dense.toarray()

            model.fit(X_train_dense, y_train)
            y_pred = model.predict(X_test_dense)
        else:
            raise e

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred
    }

    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}\n")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_model_name = name
        best_y_pred = y_pred

print(f"\n✅ Best performing model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# --- Save the Best Model ---
if best_model:
    joblib.dump(best_model, 'best_model.pkl')
    print("Best model saved as 'best_model.pkl'")

results_df = pd.DataFrame(results).T.drop(columns=['y_pred'])
results_df.to_csv('model_comparison_results.csv')
print("Model comparison results saved as 'model_comparison_results.csv'")

# =============================
# 📊 Visualization Section
# =============================

# --- 1️⃣ Model Performance Comparison ---
plt.figure(figsize=(10, 6))
results_df[['accuracy', 'precision', 'recall', 'f1_score']].plot(kind='bar')
plt.title("Model Performance Comparison", fontsize=16)
plt.ylabel("Score")
plt.xlabel("Models")
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# --- 2️⃣ Confusion Matrix for Best Model ---
cm = confusion_matrix(y_test, best_y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()

# --- 3️⃣ Classification Report Heatmap (optional advanced plot) ---
report = classification_report(y_test, best_y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(7, 5))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu")
plt.title(f"Classification Report - {best_model_name}")
plt.tight_layout()
plt.show()
