# iris_analysis.py

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Step 2: Load Dataset
print("Loading dataset...")  # Debugging line before loading the dataset

try:
    iris_data = load_iris(as_frame=True)  # Loading the Iris dataset
    df = iris_data.frame  # Assigning the data frame
    print("Dataset loaded successfully!")  # Confirmation line after successful loading
except Exception as e:
    print(f"Error loading dataset: {e}")  # Error handling for loading failure

# Step 3: Inspect Dataset
print("\nInspecting dataset...")  # Debugging line to indicate we are inspecting the dataset
print(df.head())  # Display the first 5 rows to verify data


# Step 4: Clean Dataset (if needed)
df.dropna(inplace=True)

# Step 5: Basic Statistics
print("\n📊 Summary Statistics:")
print(df.describe())

# Step 6: Group by Species and Calculate Mean
df['species'] = df['target'].map(dict(enumerate(iris_data.target_names)))
species_mean = df.groupby('species').mean()
print("\n📊 Mean values by species:")
print(species_mean)

# Step 7: Data Visualizations
sns.set(style="whitegrid")
df['index'] = df.index  # Simulate time index

# Line Chart
plt.figure(figsize=(10, 6))
sns.lineplot(x='index', y='sepal length (cm)', data=df, label='Sepal Length')
sns.lineplot(x='index', y='petal length (cm)', data=df, label='Petal Length')
plt.title('Sepal and Petal Length Over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Length (cm)')
plt.legend()
plt.savefig("line_chart.png")
plt.show()

# Bar Chart
plt.figure(figsize=(8, 5))
sns.barplot(x=species_mean.index, y=species_mean['petal length (cm)'])
plt.title('Average Petal Length per Species')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')
plt.savefig("bar_chart.png")
plt.show()

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal length (cm)'], bins=20, kde=True)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.savefig("histogram.png")
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.savefig("scatter_plot.png")
plt.show()

# Final Observations
print("\n✅ Analysis Complete. Charts saved as PNG files.")
