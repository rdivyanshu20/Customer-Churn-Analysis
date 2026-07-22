import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve

# 1. Generate Synthetic Customer Data
print("--- Generating Synthetic Dataset ---")
np.random.seed(42)
n_samples = 1000

data = {
    'CustomerID': range(1, n_samples + 1),
    'Age': np.random.randint(18, 70, n_samples),
    'Tenure_Months': np.random.randint(1, 72, n_samples),
    'Monthly_Charges': np.random.uniform(20.0, 120.0, n_samples),
    'Contract_Type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'Support_Tickets': np.random.randint(0, 10, n_samples)
}
df = pd.DataFrame(data)

# Simulate Churn based on behavior (higher tickets & lower tenure = higher churn probability)
churn_prob = (df['Support_Tickets'] * 0.1) - (df['Tenure_Months'] * 0.005)
df['Churn'] = np.where(churn_prob + np.random.normal(0, 0.2, n_samples) > 0.4, 'Yes', 'No')


# 2. Exploratory Data Analysis (EDA)
print("\n--- Performing EDA ---")
# Display basic info
print(df.info())
print("\nTarget Variable Distribution:\n", df['Churn'].value_counts())

# Plotting Churn Count
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title('Customer Churn Distribution')
plt.show()


# 3. Data Preprocessing & Cleaning

print("\n--- Preprocessing Data ---")
# Drop irrelevant features (Feature Selection)
df_clean = df.drop(['CustomerID'], axis=1)

# Encode categorical variables
le = LabelEncoder()
df_clean['Churn'] = le.fit_transform(df_clean['Churn']) # Yes=1, No=0
df_clean = pd.get_dummies(df_clean, columns=['Contract_Type'], drop_first=True)

# Separate features (X) and target (y)
X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn']

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Classification Models

print("\n--- Training Models ---")

# Model 1: Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)
log_probs = log_model.predict_proba(X_test_scaled)[:, 1]

# Model 2: Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train) # Trees don't require scaling
tree_preds = tree_model.predict(X_test)
tree_probs = tree_model.predict_proba(X_test)[:, 1]

# 5. Model Evaluation (Accuracy, ROC-AUC)

print("\n--- Logistic Regression Results ---")
print(f"Accuracy: {accuracy_score(y_test, log_preds):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, log_probs):.4f}")
print("Classification Report:\n", classification_report(y_test, log_preds))

print("\n--- Decision Tree Results ---")
print(f"Accuracy: {accuracy_score(y_test, tree_preds):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, tree_probs):.4f}")
print("Classification Report:\n", classification_report(y_test, tree_preds))