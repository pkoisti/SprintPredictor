# Import required libraries for machine learning and visualization
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

class SprintSuccessModel:
    """
    A machine learning model that predicts whether a task will be completed in a sprint.
    """
    def __init__(self, model_type='random_forest'):
        """
        Initialize the model with specified type.
        Args:
            model_type (str): Either 'random_forest', 'xgboost', 'mlp', or 'lightgbm'
        """
        self.model_type = model_type
        
        # Define which columns contain text data (like task type and assignee)
        self.categorical = ["issue_type", "assignee"]
        
        # Create base classifier based on model type
        if model_type == 'random_forest':
            base_classifier = RandomForestClassifier(
                n_estimators=100, 
                random_state=42
            )
        elif model_type == 'xgboost':
            base_classifier = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'mlp':
            base_classifier = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )
        elif model_type == 'lightgbm':
            base_classifier = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=-1  # Suppress LightGBM output
            )
        else:
            raise ValueError("model_type must be either 'random_forest', 'xgboost', 'mlp', or 'lightgbm'")
        
        # Create a pipeline that:
        # 1. Converts text data to numbers (OneHotEncoder)
        # 2. Scales numerical data (StandardScaler)
        # 3. Makes predictions (RandomForestClassifier, XGBoost, MLP, or LightGBM)
        self.model = Pipeline([
            ("preprocess", ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical)
            ], remainder='passthrough')),
            ("scaler", StandardScaler(with_mean=False)),  # Don't center sparse matrices
            ("classifier", base_classifier)
        ])

    def train(self, X, y):
        """Train the model on the provided data."""
        self.model.fit(X, y)
        
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates for each class."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        # Get feature names after preprocessing
        feature_names = (
            self.model.named_steps['preprocess']
            .named_transformers_['cat']
            .get_feature_names_out(self.categorical)
        )
        # Add numerical feature names
        numerical_features = [col for col in X.columns if col not in self.categorical]
        all_features = np.concatenate([feature_names, numerical_features])
        
        # Get importance scores
        importances = self.model.named_steps['classifier'].feature_importances_
        
        return pd.Series(importances, index=all_features)
    
    def plot_confusion_matrix(self, y_true, y_pred, ax=None):
        """Plot a confusion matrix for model evaluation."""
        cm = confusion_matrix(y_true, y_pred)
        if ax is None:
            plt.figure(figsize=(6, 4))
            ax = plt.gca()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Not Completed", "Completed"],
                    yticklabels=["Not Completed", "Completed"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix: Sprint Success ({self.model_type})")
        if ax is None:
            plt.tight_layout()
            plt.show()

    def evaluate(self, X, y):
        """Show how well the model performs"""
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Print detailed performance metrics
        report = classification_report(y, y_pred, zero_division=0)
        print(report)
        
        # Create a confusion matrix visualization
        self.plot_confusion_matrix(y, y_pred)

    def save(self, path):
        """Save the trained model to a file"""
        joblib.dump(self.model, path)

    def load(self, path):
        """Load a trained model from a file"""
        self.model = joblib.load(path) 
