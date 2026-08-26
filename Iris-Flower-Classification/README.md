# 🌸 Iris Flower Classification

## 📌 Project Overview

Iris Flower Classification is a Machine Learning project that predicts the species of an Iris flower based on its sepal and petal measurements.

The project uses a Random Forest Classifier and provides an interactive web interface built with Streamlit.

## 🎯 Objective

The objective of this project is to build a Machine Learning classification model that can identify an Iris flower as:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

## 📊 Dataset

The dataset contains 150 Iris flower samples with the following features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

There are three target classes:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## 🤖 Machine Learning Model

A Random Forest Classifier is used for classification.

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

### Model Accuracy

The trained model achieved:

**100% accuracy on the test dataset.**

## 📈 Data Visualization

Seaborn Pairplot was used to visualize the relationships between the different flower measurements and species.

## 🌐 Streamlit Web App

The project includes an interactive Streamlit application where users can enter:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The application then predicts the Iris flower species.

## 📂 Project Structure

```text
iris-flower-classification/
│
├── iris.csv
├── iris_model.pkl
├── train_model.py
├── app.py
├── requirements.txt
└── README.md