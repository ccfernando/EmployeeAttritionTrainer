# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace with the actual path to your dataset)
data = pd.read_csv("data/train.csv")

# Set the style for seaborn plots
sns.set(style="whitegrid")

# Create a smaller figure for the plots
plt.figure(figsize=(16, 12))  # Smaller figure to avoid overlap

# Define the custom color palette
custom_palette = {0: 'blue', 1: 'red'}

# Plot 1: Distribution of Age by Attrition
plt.subplot(3, 2, 1)
sns.histplot(data=data, x='Age', hue='Attrition', kde=True, multiple="stack", bins=20, palette=custom_palette)
plt.title('Distribution of Age by Attrition', fontsize=12)

# Plot 2: Distribution of Monthly Income by Attrition
plt.subplot(3, 2, 2)
sns.histplot(data=data, x='MonthlyIncome', hue='Attrition', kde=True, multiple="stack", bins=20, palette=custom_palette)
plt.title('Distribution of Monthly Income by Attrition', fontsize=12)

# Plot 3: Attrition based on Marital Status
plt.subplot(3, 2, 3)
sns.countplot(x='MaritalStatus', data=data, hue='Attrition', palette=custom_palette, legend=False)
plt.title('Attrition Based on Marital Status', fontsize=12)
# Make x-axis labels horizontal
plt.xticks(rotation=0, fontsize=10)

# Plot 4: Count of employees by Department and Attrition
plt.subplot(3, 2, 4)
sns.countplot(x='Department', data=data, hue='Attrition', palette=custom_palette, legend=False)
plt.title('Count of Employees by Department and Attrition', fontsize=12)
# Rotate x-axis labels for Department to horizontal
plt.xticks(rotation=0, fontsize=10)

# Plot 5: Count of employees by Gender and Attrition
plt.subplot(3, 2, 5)
sns.countplot(x='Gender', data=data, hue='Attrition', palette=custom_palette, legend=False)
plt.title('Count of Employees by Gender and Attrition', fontsize=12)
# Rotate x-axis labels for Gender to horizontal
plt.xticks(rotation=0, fontsize=10)

# Plot 6: Relationship Between Age and Monthly Income by Attrition
plt.subplot(3, 2, 6)
sns.scatterplot(x='Age', y='MonthlyIncome', data=data, hue='Attrition', palette=custom_palette, style='Attrition')
plt.title('Relationship Between Age and Monthly Income by Attrition', fontsize=12)

# Adjust layout to make sure everything fits
plt.tight_layout(pad=2.0)  # Reduce padding between plots

# Show the plots
plt.show()
