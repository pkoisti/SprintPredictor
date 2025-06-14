import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, 
    QTextEdit, QGroupBox, QSizePolicy, QInputDialog, QDialog, QProgressBar,
    QComboBox
)
from PyQt5.QtGui import QFont
from src.model import SprintSuccessModel
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from src.jira_client import JiraClient
from src.data_processing import process_issues_to_df
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Load environment variables from .env file
load_dotenv()

class MainWindow(QWidget):
    """
    The main window of our application.
    Shows buttons for training and prediction, and displays results.
    """
    def __init__(self):
        super().__init__()
        # Set up the window
        self.setWindowTitle('Jira Sprint Success Predictor')
        self.resize(1500, 800)
        
        # Initialize variables
        self.model = None  # Our machine learning model
        self.train_data = None  # Data used to train the model
        self.train_path = None  # Path to training data file
        self.predict_data = None  # Data we want to make predictions for
        self.predict_path = None  # Path to prediction data file
        self.jira_configured = self.check_jira_config()  # Check if we can use Jira

        # Create the main layout
        self.layout = QVBoxLayout()

        # Create buttons for CSV operations
        self.combined_controls_layout = QHBoxLayout()
        self.train_csv_button = QPushButton('Train Model from CSV')
        self.train_csv_button.clicked.connect(self.train_model_from_csv)
        self.combined_controls_layout.addWidget(self.train_csv_button)
        self.predict_csv_button = QPushButton('Predict from CSV')
        self.predict_csv_button.clicked.connect(self.predict_from_csv)
        self.combined_controls_layout.addWidget(self.predict_csv_button)
        self.layout.addLayout(self.combined_controls_layout)

        # Create buttons for Jira operations
        self.jira_controls_layout = QHBoxLayout()
        self.train_jira_button = QPushButton('Train from Jira')
        self.train_jira_button.clicked.connect(self.train_from_jira)
        self.jira_controls_layout.addWidget(self.train_jira_button)
        self.predict_jira_button = QPushButton('Predict from Jira')
        self.predict_jira_button.clicked.connect(self.predict_from_jira)
        self.jira_controls_layout.addWidget(self.predict_jira_button)
        self.layout.addLayout(self.jira_controls_layout)

        # Add a progress bar for Jira operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        # Add model selection dropdown
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Random Forest", "XGBoost", "Neural Network", "LightGBM"])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        self.layout.addLayout(model_layout)
        
        # Initialize model
        self.model = None
        self.on_model_changed()

        # Add labels to show what data is loaded
        self.info_layout = QHBoxLayout()
        self.train_label = QLabel('No training data loaded')
        self.info_layout.addWidget(self.train_label)
        self.predict_label = QLabel('No prediction data loaded')
        self.info_layout.addWidget(self.predict_label)
        self.layout.addLayout(self.info_layout)

        # Split the window into left and right panels
        self.split_layout = QHBoxLayout()

        # Left panel: Shows how well the model performs
        self.left_panel = QVBoxLayout()
        # Top: Model accuracy report
        self.metrics_box = QGroupBox('Model Accuracy')
        self.metrics_box_layout = QVBoxLayout()
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet("font-family: 'Courier New';")
        self.report_text.setVisible(False)
        self.metrics_box_layout.addWidget(self.report_text)
        self.metrics_box.setLayout(self.metrics_box_layout)
        self.left_panel.addWidget(self.metrics_box)
        # Bottom: Confusion matrix
        self.confusion_box = QGroupBox('Confusion Matrix')
        self.confusion_box_layout = QVBoxLayout()
        self.figure = plt.figure(figsize=(4, 3))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setVisible(False)
        self.confusion_box_layout.addWidget(self.canvas)
        self.confusion_box.setLayout(self.confusion_box_layout)
        self.left_panel.addWidget(self.confusion_box)
        self.split_layout.addLayout(self.left_panel, 1)

        # Right panel: Shows predictions
        self.right_panel = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_panel.addWidget(self.table)
        self.split_layout.addLayout(self.right_panel, 2)

        self.layout.addLayout(self.split_layout)
        self.setLayout(self.layout)

    def on_model_changed(self):
        """Handle model type change."""
        model_map = {
            "Random Forest": "random_forest",
            "XGBoost": "xgboost",
            "Neural Network": "mlp",
            "LightGBM": "lightgbm"
        }
        model_type = model_map[self.model_combo.currentText()]
        self.model = SprintSuccessModel(model_type=model_type)
        if hasattr(self, 'X') and hasattr(self, 'y'):
            self.train_model()  # Retrain with new model type

    def check_jira_config(self):
        """Check if we have all the required Jira login information"""
        required_vars = ["JIRA_DOMAIN", "JIRA_EMAIL", "JIRA_API_TOKEN"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            return False
        return True

    def show_jira_config_error(self):
        """Show an error message if Jira login information is missing"""
        message = """Jira credentials are not configured. Please set the following environment variables:
- JIRA_DOMAIN
- JIRA_EMAIL
- JIRA_API_TOKEN

You can set these in a .env file or in your system environment variables."""
        self.show_selectable_dialog('Jira Configuration Error', message)

    def train_model_from_csv(self):
        """Let the user select a CSV file to train the model"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Training CSV File", "data", "CSV Files (*.csv)")
        if file_path:
            try:
                df = pd.read_csv(file_path)
                self.train_data = df
                self.train_path = file_path
                self.train_label.setText(f'Training data: {os.path.basename(file_path)}')
                self.train_model()

            except Exception as e:
                self.show_selectable_dialog('Error', f'Failed to load CSV: {e}')
    
    def train_model(self):
        """Train the machine learning model with the loaded data"""
        if self.train_data is not None:
            df = self.train_data.copy()
            if 'sprint_success' not in df.columns:
                self.show_selectable_dialog('Error', 'Training CSV must contain a "sprint_success" column.')
                return
            
            # Use only the columns the model needs
            model_columns = [
                "issue_type", "assignee", "original_estimate", "was_in_previous_sprint",
                "days_in_sprint", "comment_count", "tasks_per_assignee", "sprint_success"
            ]
            df = df[model_columns]
            
            # Split data into training and testing sets
            X = df.drop(columns='sprint_success')
            y = df['sprint_success']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            
            # Create and train the model
            self.model.train(X_train, y_train)
            
            # Show how well the model performs
            self.show_accuracy(X_test, y_test)
            self.show_selectable_dialog('Model Trained', 'Model training complete!')

    def predict_from_csv(self):
        """Let the user select a CSV file to make predictions"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Prediction CSV File", "data", "CSV Files (*.csv)")
        if file_path:
            try:
                df = pd.read_csv(file_path)
                self.predict_data = df
                self.predict_path = file_path
                self.predict_label.setText(f'Prediction data: {os.path.basename(file_path)}')
                self.predict()
 
            except Exception as e:
                self.show_selectable_dialog('Error', f'Failed to load CSV: {e}')

    def predict(self):
        """Make predictions using the trained model"""
        if self.model is None:
            self.show_selectable_dialog('Error', 'Train the model first!')
            return
        if self.predict_data is not None:
            df = self.predict_data.copy()
            X = df.drop(columns=['sprint_success'], errors='ignore')
            predictions = self.model.predict(X)
            self.show_predictions(df, predictions)
    
    def show_predictions(self, df, predictions):
        """Show the predictions in a table"""
        if len(predictions) != len(df):
            self.show_selectable_dialog('Prediction Error', f'Number of predictions ({len(predictions)}) does not match number of rows ({len(df)}).')
            return
        
        # Set up the table columns
        display_columns = ['Issue Type', 'Key', 'Summary', 'Predicted Outcome']
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(display_columns))
        self.table.setHorizontalHeaderLabels(display_columns)
        
        # Map column names from our data to display names
        column_mapping = {
            'Issue Type': 'issue_type',
            'Key': 'key',
            'Summary': 'summary'
        }
        
        # Fill in the table with predictions
        for i, row in df.iterrows():
            # Set issue type
            self.table.setItem(i, 0, QTableWidgetItem(str(row['issue_type'])))
            # Set key
            self.table.setItem(i, 1, QTableWidgetItem(str(row['key'])))
            # Set summary
            self.table.setItem(i, 2, QTableWidgetItem(str(row['summary'])))
            # Set predicted outcome with icon
            icon = '✅' if predictions[i] == 1 else '❌'
            self.table.setItem(i, 3, QTableWidgetItem(f"{icon} {'Will Complete' if predictions[i] == 1 else 'Will Not Complete'}"))
        
        # Make the table look nice
        self.table.resizeColumnsToContents()
        self.table.setVisible(True)

    def show_accuracy(self, X, y):
        """
        Shows how well our model performs on test data.
        Creates a report showing accuracy and a confusion matrix.
        """
        from sklearn.metrics import classification_report, confusion_matrix
        
        #Classification report
        y_pred = self.model.predict(X)
        report = classification_report(y, y_pred, zero_division=0)
        self.report_text.setText(report)
        self.report_text.setVisible(True)
        
        #Confusion matrix
        cm = confusion_matrix(y, y_pred)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                    xticklabels=["Not Completed", "Completed"], 
                    yticklabels=["Not Completed", "Completed"])
        ax.set_xlabel("Predicted", fontsize=9, labelpad=4)
        ax.set_ylabel("Actual", fontsize=9, labelpad=4)
        ax.set_title("Confusion Matrix: Sprint Success", fontsize=10, pad=4)
        self.figure.tight_layout()
        self.canvas.draw()
        self.canvas.setVisible(True)

    def show_selectable_dialog(self, title, message):
        """
        Shows a popup window with a message that can be copied.
        Used to show errors or important information to the user.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        layout = QVBoxLayout()
        text = QTextEdit()
        text.setReadOnly(True)
        text.setText(message)
        text.setMinimumWidth(400)
        text.setMinimumHeight(120)
        layout.addWidget(text)
        btn = QPushButton('OK')
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.setLayout(layout)
        dlg.exec_()

    def train_from_jira(self):
        """
        Trains the model using data from Jira.
        Lets the user select a board and number of sprints to use for training.
        """
        # Check if we have Jira login information
        if not self.jira_configured:
            self.show_jira_config_error()
            return
        try:
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            QApplication.processEvents()
            client = JiraClient()
            boards = client.get_boards()
            if not boards:
                self.progress_bar.setVisible(False)
                self.show_selectable_dialog('Jira', 'No scrum boards found for the project key.')
                return
            board_names = [f"{b['name']} (id: {b['id']})" for b in boards]
            board_ids = [b['id'] for b in boards]
            board_idx, ok = QInputDialog.getItem(self, 'Select Board', 'Choose a Jira scrum board:', board_names, 0, False)
            if not ok:
                self.progress_bar.setVisible(False)
                return
            selected_board_id = board_ids[board_names.index(board_idx)]
            sprints = client.get_sprints(board_id=selected_board_id, count=100)
            closed_sprints = [s for s in sprints if s.get('state') == 'closed']
            all_parsed_issues = []
            total_sprints = len(closed_sprints)
            
            for idx, sprint in enumerate(closed_sprints):
                self.progress_bar.setValue(int((idx + 1) / total_sprints * 100))
                sprint_id = sprint['id']
                sprint_end = pd.to_datetime(sprint.get('endDate')) if sprint.get('endDate') else None
                issues = client.get_issues_for_sprint(sprint_id)
                for issue in issues:
                    parsed = client.parse_issue(issue, sprint_end)
                    parsed['sprint_id'] = sprint_id
                    all_parsed_issues.append(parsed)
            df = process_issues_to_df(all_parsed_issues)
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)
            if df.empty:
                self.show_selectable_dialog('Jira', 'No usable issues found in closed sprints.')
                return
            self.train_data = df
            self.train_label.setText(f'Training data: {board_idx} from Jira closed sprints')
            self.train_model()

            default_path = os.path.join('data', 'jira_sprint_data.csv')
            path, _ = QFileDialog.getSaveFileName(self, 'Save as CSV?', default_path, 'CSV Files (*.csv)')
            if path:
                df.to_csv(path, index=False)
                self.show_selectable_dialog('Saved', f'Data saved to {path}')
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.show_selectable_dialog('Jira Error', str(e))

    def predict_from_jira(self):
        """
        Makes predictions for tasks in a current Jira sprint.
        Lets the user select a board and sprint to predict for.
        """
        # Check if we have Jira login information
        if not self.jira_configured:
            self.show_jira_config_error()
            return
        
        try:
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            QApplication.processEvents()
            client = JiraClient()
            boards = client.get_boards()
            if not boards:
                self.progress_bar.setVisible(False)
                self.show_selectable_dialog('Jira', 'No scrum boards found for the project key.')
                return
            board_names = [f"{b['name']} (id: {b['id']})" for b in boards]
            board_ids = [b['id'] for b in boards]
            board_idx, ok = QInputDialog.getItem(self, 'Select Board', 'Choose a Jira scrum board:', board_names, 0, False)
            if not ok:
                self.progress_bar.setVisible(False)
                return
            selected_board_id = board_ids[board_names.index(board_idx)]
            open_sprints = client.get_open_sprints(board_id=selected_board_id)
            if not open_sprints:
                self.progress_bar.setVisible(False)
                self.show_selectable_dialog('Jira', 'No open or backlog sprints found for prediction.')
                return
            sprint_names = [f"{s['name']} (id: {s['id']})" for s in open_sprints]
            sprint_ids = [s['id'] for s in open_sprints]
            sprint_idx, ok = QInputDialog.getItem(self, 'Select Sprint', 'Choose an open/backlog sprint for prediction:', sprint_names, 0, False)
            if not ok:
                self.progress_bar.setVisible(False)
                return
            selected_sprint_id = sprint_ids[sprint_names.index(sprint_idx)]
            issues = client.get_issues_for_sprint(selected_sprint_id)
            sprint_end = None
            for s in open_sprints:
                if s['id'] == selected_sprint_id:
                    sprint_end = pd.to_datetime(s.get('endDate')) if s.get('endDate') else None
                    break
            parsed_issues = []
            for issue in issues:
                parsed = client.parse_issue(issue, sprint_end)
                parsed['sprint_id'] = selected_sprint_id
                parsed_issues.append(parsed)
            df = process_issues_to_df(parsed_issues)
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)
            if df.empty:
                self.show_selectable_dialog('Jira', 'No usable issues found for the selected sprint.')
                return
            self.predict_data = df
            self.predict_label.setText(f'Prediction data: {sprint_idx} from Jira open/backlog sprint')
            self.predict()

            default_path = os.path.join('data', 'jira_sprint_data.csv')
            path, _ = QFileDialog.getSaveFileName(self, 'Save as CSV?', default_path, 'CSV Files (*.csv)')
            if path:
                df.to_csv(path, index=False)
                self.show_selectable_dialog('Saved', f'Data saved to {path}')
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.show_selectable_dialog('Jira Error', str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 