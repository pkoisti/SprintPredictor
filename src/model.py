import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class SprintSuccessModel:
    """
    Random Forest (Logistic Regression for now) model for sprint success prediction.
    """
    def __init__(self):
        self.categorical = ["issue_type", "assignee"]
        self.model = Pipeline([
            ("preprocess", ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical)
            ], remainder='passthrough')),
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", LogisticRegression(max_iter=5000))
        ])

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        report = classification_report(y, y_pred, zero_division=0)
        print(report)
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Completed", "Completed"], yticklabels=["Not Completed", "Completed"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix: Sprint Success")
        plt.tight_layout()
        plt.show()

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path) 