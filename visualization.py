# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace with the actual path to your dataset)
data = pd.read_csv("data/train.csv")

# Set the style for seaborn plots
sns.set(style="whitegrid")

# Create a larger figure for the plots
plt.figure(figsize=(24, 18))  # Larger figure to avoid overlap

# Plot 1: Distribution of Age
plt.subplot(3, 2, 1)
sns.histplot(data['Age'], kde=True, color='blue', bins=20)
plt.title('Distribution of Age', fontsize=14)

# Plot 2: Distribution of Monthly Income
plt.subplot(3, 2, 2)
sns.histplot(data['MonthlyIncome'], kde=True, color='green', bins=20)
plt.title('Distribution of Monthly Income', fontsize=14)

# Plot 3: Count of Attrition (Yes/No)
plt.subplot(3, 2, 3)
sns.countplot(x='Attrition', data=data, hue='Attrition', legend=False)
plt.title('Count of Attrition (Yes/No)', fontsize=14)
plt.xticks(fontsize=12)

# Plot 4: Count of employees by Department
plt.subplot(3, 2, 4)
sns.countplot(x='Department', data=data, hue='Department', legend=False)
plt.title('Count of Employees by Department', fontsize=14)
# Rotate x-axis labels for Department to 90 degrees to prevent overlap
plt.xticks(rotation=90, fontsize=12)

# Plot 5: Count of employees by Gender
plt.subplot(3, 2, 5)
sns.countplot(x='Gender', data=data, hue='Gender', legend=False)
plt.title('Count of Employees by Gender', fontsize=14)
# Rotate x-axis labels for Gender to 90 degrees
plt.xticks(rotation=90, fontsize=12)

# Plot 6: Relationship between MonthlyIncome and Age
plt.subplot(3, 2, 6)
sns.scatterplot(x='Age', y='MonthlyIncome', data=data, hue='Attrition', palette='Set1')
plt.title('Relationship Between Age and Monthly Income', fontsize=14)

# Adjust layout to make sure everything fits
plt.tight_layout(pad=3.0)  # Increase padding between plots

# Show the plots
plt.show()
