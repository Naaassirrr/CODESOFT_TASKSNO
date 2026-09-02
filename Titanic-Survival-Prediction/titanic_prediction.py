import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())
print(df.info())

print(df.isnull().sum())

print("Median Age:",df["Age"].median())

df["Age"]=df["Age"].fillna(df["Age"].median())

features = ["Pclass","Sex","Age","Fare"]
X=df[features]
y=df["Survived"]

X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

import matplotlib.pyplot as plt

survival_count = df["Survived"].value_counts()

print(survival_count)

survival_count.plot(kind="bar")

plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.savefig("survival_count.png")
plt.close()

gender_survival = df.groupby("Sex")["Survived"].mean()

print(gender_survival)

gender_survival.plot(kind="bar")

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

plt.savefig("survival_by_gender.png")
plt.close()

class_survival = df.groupby("Pclass")["Survived"].mean()

print(class_survival)

class_survival.plot(kind="bar")

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.savefig("survival_by_class.png")
plt.close()

age_data = df["Age"]

age_data.plot(kind="hist", bins=20)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.savefig("age_distribution.png")
plt.close()