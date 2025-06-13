# Jira Sprint Success Predictor

Final project for the Building AI course

## Summary

A machine learning application that predicts whether Jira issues will be completed within a sprint. Using historical sprint data and AI, it helps teams make better sprint planning decisions. The application includes a demo mode that works with CSV files, making it easy to explore without requiring Jira credentials.

## Background

Sprint planning in agile teams often faces several challenges:
* Difficulty in estimating which issues can be completed within a sprint
* Overcommitting to too many tasks
* Not accounting for historical performance patterns
* Lack of data-driven decision making in sprint planning

This project aims to help teams make better sprint planning decisions by using AI to predict task completion based on historical data.

## How is it used?

The application can be used in two ways:

1. **Demo Mode (Recommended)** - Works with CSV files:
   - Use the included `mock_dev.csv` file to try out the application
   - Train the model and make predictions using CSV files
   - View predictions and model performance metrics
   - No Jira setup required

2. **Jira Integration Mode** - For teams using Jira:
   - Train the model with your team's historical sprint data
   - Get predictions for new sprint issues
   - Requires Jira credentials and API access

## Features

- 🤖 Machine learning model for sprint success prediction
- 📊 Visual model evaluation with confusion matrix
- 📈 Sprint completion predictions
- 🎯 Issue-level success probability
- 🖥️ Desktop GUI application
- 🔄 Optional Jira integration

## Data and AI

The application uses two types of data:
1. CSV files (primary method)
   - Use the included `mock_dev.csv` for testing
   - Create your own CSV files with the same format
2. Jira data (optional)
   - Requires Jira credentials
   - Fetches data from your Jira instance

The AI model uses the following features:

| Feature | Description |
| ------- | ----------- |
| Original Estimate | Initial time estimate for the task |
| Sprint History | Whether the task was in a previous sprint |
| Issue Type | Type of the task (Story, Bug, etc.) |
| Assignee Load | Number of tasks per assignee |
| Sprint Duration | Days allocated in sprint |
| Activity Level | Comment count |

The core of the prediction engine is a Random Forest Classifier from scikit-learn, chosen for its:
- Ability to work with both numbers (like time estimates) and categories (like issue types)
- Clear explanation of which factors most influence sprint success
- Good performance even with small amounts of historical data
- Reliable predictions that don't overfit to past patterns

## Challenges

The project has some limitations and ethical considerations:
* Requires sufficient historical data for accurate predictions
* May reinforce existing biases in estimation patterns
* Cannot account for unexpected external factors affecting sprint completion
* Does not replace human judgment in sprint planning
* Could potentially be misused to create unrealistic performance expectations
* Limited by the quality and consistency of input data

## What next?

Future development possibilities:
1. Improve prediction accuracy
   - Add more features from Jira data
   - Try different machine learning models
   - Include team velocity in predictions

2. Enhance user experience
   - Add visualizations for prediction confidence
   - Show historical prediction accuracy
   - Create a web interface

3. Expand functionality
   - Support more project management tools
   - Add team capacity predictions
   - Include natural language analysis of issue descriptions

4. Make it more accessible
   - Create a web-based version
   - Add API for integration with other tools
   - Provide more documentation and examples

Skills needed: Deep learning expertise, web development, UX design

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- PyQt5 for the desktop interface

### Quick Start (Demo Mode)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jira-sprint-predictor.git
cd jira-sprint-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python desktop_app.py
```

5. Use the "Train Model from CSV" button to load `mock_dev.csv`
6. Use the "Predict from CSV" button to make predictions

### Optional: Jira Integration

To use with your Jira instance, create a `.env` file with your credentials:
```
JIRA_DOMAIN=your-domain.atlassian.net
JIRA_EMAIL=your-email@company.com
JIRA_API_TOKEN=your-api-token
JIRA_PROJECT_KEY=YOUR_PROJECT  # Optional: limit to specific project
```

## Project Structure

```
jira-sprint-predictor/
├── data/                  # Data directory
│   └── mock_dev.csv      # Demo data file
├── src/                  # Source code
│   ├── data_processing.py
│   ├── jira_client.py
│   └── model.py
├── desktop_app.py        # Main application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Security

- No sensitive data is stored in the repository
- Jira credentials are stored in `.env` (not committed)
- Training data and models are excluded from git
- All API calls use secure authentication

## Acknowledgments

* Building AI course by Reaktor Innovations and University of Helsinki

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 