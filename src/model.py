# Import required libraries for machine learning and visualization
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
    A machine learning model that predicts whether a task will be completed in a sprint.
    Uses Logistic Regression with preprocessing for both numerical and categorical data.
    """
    def __init__(self):
        # Define which columns contain text data (like task type and assignee)
        self.categorical = ["issue_type", "assignee"]
        
        # Create a pipeline that:
        # 1. Converts text data to numbers (OneHotEncoder)
        # 2. Scales numerical data (StandardScaler)
        # 3. Makes predictions (LogisticRegression)
        self.model = Pipeline([
            ("preprocess", ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical)
            ], remainder='passthrough')),
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", LogisticRegression(max_iter=5000))
        ])

    def train(self, X, y):
        """Train the model using historical sprint data"""
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions for new sprint tasks"""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Show how well the model performs"""
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Print detailed performance metrics
        report = classification_report(y, y_pred, zero_division=0)
        print(report)
        
        # Create a confusion matrix visualization
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=["Not Completed", "Completed"], 
                   yticklabels=["Not Completed", "Completed"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix: Sprint Success")
        plt.tight_layout()
        plt.show()

    def save(self, path):
        """Save the trained model to a file"""
        joblib.dump(self.model, path)

    def load(self, path):
        """Load a trained model from a file"""
        self.model = joblib.load(path) 