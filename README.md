# Jira Sprint Success Predictor

Building AI course project

## Summary

A machine learning application that predicts whether planned Jira issues can be completed within a sprint. Using historical sprint data and AI, it helps teams make better sprint planning decisions. The application includes a demo mode that works with CSV files, making it easy to explore without requiring Jira credentials.

## Background

Sprint planning in agile teams is a collaborative meeting to decide which tasks to complete in the upcoming sprint based on team capacity and priorities.

However, this critical step often faces several challenges:
* Difficulty in estimating which issues are realistically achievable within the sprint
* Overcommitting to too many tasks
* Failing to account for historical team performance

 I’ve seen firsthand how sprint planning often leads to overcommitment and missed deadlines. Jira, being one of the most widely used tools for tracking Agile development, provides rich data on issues, assignees, time estimates, and sprint history. This project uses that data with AI to predict task completion likelihood, helping teams plan more realistically and sustainably.

## How is it used?

The application can be used in two ways:

1. **Demo** - Works with CSV files:
   - Use the included `mock_data.csv` file to try out the application
   - Train the model and make predictions using CSV files
   - View predictions and model performance metrics
   - No Jira setup required

2. **Jira Integration** - For teams using Jira:
   - Train the model with your team's historical sprint data
   - Get predictions for new sprint issues
   - Requires Jira credentials and API access

<img src="assets/screenshot.png" alt="Screenshot of Sprint Success Predictor app interface" width="800"/>

## Features

- 🤖 Machine learning model for sprint success prediction
- 📊 Visual model evaluation with confusion matrix
- 🎯 Issue-level sprint success prediction (yes/no outcome)
- 🖥️ Desktop GUI application
- 🔄 Optional Jira integration

## Data and AI techniques

The provided application uses two types of data:
1. CSV files
   - Use the included `mock_data.csv` for testing
   - Create your own CSV files with the same format
2. Jira data (optional)
   - Requires Jira credentials
   - Fetches data from your Jira instance
   - Option to save fetched data into CSV for later use

The AI model uses the following features:

| Feature | Description |
| ------- | ----------- |
| Original Estimate | Initial time estimate for the task |
| Sprint History | Whether the task was in a previous sprint |
| Issue Type | Type of the task (Story, Bug, etc.) |
| Assignee Load | Number of tasks per assignee |
| Issue Age at Sprint End | Days between issue creation and sprint end |
| Activity Level | Comment count |

The core of the prediction engine is a Random Forest Classifier from scikit-learn, chosen for its:
- Ability to handle both numerical and categorical features
- Feature importance insights
- Robust performance with limited data
- Resistance to overfitting

## Challenges

This project does not account for unexpected disruptions such as team illness, changing priorities, or external dependencies. It relies on historical data, which may reinforce existing biases or outdated practices. The model cannot assess task complexity or human factors like motivation or collaboration. It supports, but does not replace, human judgment in sprint planning. Its effectiveness also depends on the quality and consistency of Jira data.

## What next?

Future development possibilities:
1. Improve prediction accuracy
   - Add more features from Jira data
   - Try different machine learning models

2. Expand functionality
   - Support more project management tools
   - Add team capacity predictions
   - Include natural language analysis of issue descriptions

3. Make it more accessible
   - Create a web-based version
   - Add API for integration with other tools
   - Provide more documentation and examples

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- PyQt5 for the desktop interface

### Quick Start 

1. Clone the repository:
```bash
git https://github.com/paulakois/SprintPredictor.git
cd SprintPredictor
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

5. Use the "Train Model from CSV" button to load `mock_data.csv`
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
SprintPredictor/
├── data/                  # Data directory
│   └── mock_data.csv      # Demo data file
├── src/                  # Source code
│   ├── data_processing.py
│   ├── jira_client.py
│   └── model.py
├── desktop_app.py        # Main application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Security

- Jira credentials are stored in `.env` (not committed)
- Jira training data and models are excluded from git
- All API calls use secure authentication

## Acknowledgments

* Building AI course by MinnaLearn and University of Helsinki

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 