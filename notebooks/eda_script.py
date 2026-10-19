import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("screenshots", exist_ok=True)
df = pd.read_csv("data/heart_disease.csv")

# 1. Target Class Balance
plt.figure(figsize=(6, 4))
sns.countplot(x='num', data=df, palette='Set2')
plt.title("Heart Disease Target Distribution (0: Absent, 1: Present)")
plt.xlabel("Target (num)")
plt.ylabel("Patient Count")
plt.savefig("screenshots/eda_class_balance.png", bbox_inches='tight')
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("screenshots/eda_correlation_heatmap.png", bbox_inches='tight')
plt.close()

# 3. Continuous feature distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for i, feature in enumerate(features):
    row, col = divmod(i, 3)
    sns.histplot(data=df, x=feature, hue='num', kde=True, ax=axes[row, col], palette='coolwarm')
    axes[row, col].set_title(f"{feature} Distribution")
fig.delaxes(axes[1, 2])
plt.tight_layout()
plt.savefig("screenshots/eda_feature_distributions.png", bbox_inches='tight')
plt.close()

print("EDA plots generated in screenshots/")