import pandas as pd

df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

print(df.head())
print(df.info())
print(df.isnull().sum())

# Fill missing values

df["Year"] = df["Year"].str.extract(r"(\d{4})")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

df["Duration"] = df["Duration"].str.extract(r"(\d+)")
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

df["Year"] = df["Year"].fillna(df["Year"].median())
df["Duration"] = df["Duration"].fillna(df["Duration"].median())

df["Rating"] = df["Rating"].fillna(df["Rating"].median())

df["Votes"] = df["Votes"].str.replace(",", "", regex=False)
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")
df["Votes"] = df["Votes"].fillna(df["Votes"].median())

df["Genre"] = df["Genre"].fillna("Unknown")
df["Director"] = df["Director"].fillna("Unknown")
df["Actor 1"] = df["Actor 1"].fillna("Unknown")
df["Actor 2"] = df["Actor 2"].fillna("Unknown")
df["Actor 3"] = df["Actor 3"].fillna("Unknown")

print(df.isnull().sum())

# Select useful features
features = ["Year", "Duration", "Votes", "Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]

X = df[features]
y = df["Rating"]

print(X.head())
print(y.head())

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Categorical columns
categorical_features = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]

# Numerical columns
numerical_features = ["Year", "Duration", "Votes"]

# Apply One-Hot Encoding to categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

X_encoded = preprocessor.fit_transform(X)

print("Encoded data shape:", X_encoded.shape)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.ensemble import RandomForestRegressor

# Create the model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)

print("Model training completed!")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.show()

# Get feature names after encoding
feature_names = preprocessor.get_feature_names_out()

# Get feature importance from Random Forest
importances = model.feature_importances_

# Create a DataFrame
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display top 15 important features
print(feature_importance_df.head(15))

from sklearn.linear_model import LinearRegression

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Prediction
lr_pred = lr_model.predict(X_test)

# Evaluation
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("Linear Regression")
print("MAE :", lr_mae)
print("RMSE:", lr_rmse)
print("R²  :", lr_r2)

print("\nRandom Forest")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

def predict_movie_rating(year, duration, votes, genre, director,
                         actor1, actor2, actor3):

    # Create input data
    new_movie = pd.DataFrame({
        "Year": [year],
        "Duration": [duration],
        "Votes": [votes],
        "Genre": [genre],
        "Director": [director],
        "Actor 1": [actor1],
        "Actor 2": [actor2],
        "Actor 3": [actor3]
    })

    # Transform input using the same preprocessor
    new_movie_encoded = preprocessor.transform(new_movie)

    # Predict rating using Random Forest
    predicted_rating = model.predict(new_movie_encoded)

    return predicted_rating[0]


# Example movie
prediction = predict_movie_rating(
    year=2020,
    duration=120,
    votes=100,
    genre="Drama",
    director="Unknown",
    actor1="Unknown",
    actor2="Unknown",
    actor3="Unknown"
)

print("Predicted Movie Rating:", round(prediction, 2))

import joblib

# Save the trained model
joblib.dump(model, "movie_rating_model.pkl")

# Save the preprocessor
joblib.dump(preprocessor, "movie_rating_preprocessor.pkl")

print("Model saved successfully!")
print("Preprocessor saved successfully!")

import os

print("Model file exists:", os.path.exists("movie_rating_model.pkl"))
print("Preprocessor file exists:", os.path.exists("movie_rating_preprocessor.pkl"))

