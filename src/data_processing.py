import pandas as pd

def process_issues_to_df(parsed_issues):
    """
    Convert a list of Jira issues into a format suitable for the machine learning model.
    Adds useful features like whether a task was in a previous sprint.
    """
    # Convert list of issues to a DataFrame
    df = pd.DataFrame(parsed_issues)
    if df.empty:
        return df

    # Add new features that help predict sprint success
    # 1. Was this task in a previous sprint?
    df["was_in_previous_sprint"] = df.duplicated(subset=["key"], keep=False).astype(int)
    
    # 2. How many tasks are in this sprint?
    df["total_tasks_in_sprint"] = df.groupby("sprint_id")["key"].transform("count")
    
    # 3. How many different people are assigned tasks?
    df["unique_assignees_in_sprint"] = df.groupby("sprint_id")["assignee"].transform("nunique")
    
    # 4. How many tasks per person?
    df["tasks_per_assignee"] = df["total_tasks_in_sprint"] / df["unique_assignees_in_sprint"]
    
    # Handle missing data
    if 'original_estimate' in df.columns:
        df['original_estimate'] = df['original_estimate'].fillna(0)
    if 'assignee' in df.columns:
        df['assignee'] = df['assignee'].fillna('Unassigned')

    # Define which columns are needed for the model
    model_columns = [
        "issue_type", "assignee", "original_estimate", "was_in_previous_sprint",
        "days_in_sprint", "comment_count", "tasks_per_assignee", "sprint_success"
    ]
    
    # Define which columns to show in the results
    display_columns = ["key", "summary"]
    
    # Combine all needed columns
    all_columns = model_columns + display_columns
    
    # Clean the data: keep only needed columns and remove rows with missing data
    df_clean = df[all_columns].copy()
    df_clean = df_clean.dropna(subset=model_columns)
    
    return df_clean 