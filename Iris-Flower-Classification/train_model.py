import pandas as pd

# Load dataset
df = pd.read_csv("iris.csv")

# Display first 5 rows
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

import matplotlib.pyplot as plt
import seaborn as sns

# Pairplot
sns.pairplot(df, hue="species")
plt.show()

from sklearn.model_selection import train_test_split

# Features
X = df[[
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]]

# Target
y = df["species"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

# Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

from sklearn.metrics import accuracy_score

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

import pickle

# Save the trained model
with open("iris_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")