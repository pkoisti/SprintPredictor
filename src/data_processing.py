import pandas as pd

def process_issues_to_df(parsed_issues):
    """
    Convert a list of parsed issues to a DataFrame and add derived columns.
    """
    df = pd.DataFrame(parsed_issues)
    if df.empty:
        return df
    # Derived columns
    df["was_in_previous_sprint"] = df.duplicated(subset=["key"], keep=False).astype(int)
    df["total_tasks_in_sprint"] = df.groupby("sprint_id")["key"].transform("count")
    df["unique_assignees_in_sprint"] = df.groupby("sprint_id")["assignee"].transform("nunique")
    df["tasks_per_assignee"] = df["total_tasks_in_sprint"] / df["unique_assignees_in_sprint"]
    
    # Fill missing original_estimate with 0
    if 'original_estimate' in df.columns:
        df['original_estimate'] = df['original_estimate'].fillna(0)
    
    # Fill missing assignee with 'Unassigned'
    if 'assignee' in df.columns:
        df['assignee'] = df['assignee'].fillna('Unassigned')

    # Columns needed for model training
    model_columns = [
        "issue_type", "assignee", "original_estimate", "was_in_previous_sprint",
        "days_in_sprint", "comment_count", "tasks_per_assignee", "sprint_success"
    ]
    
    # Additional display columns (excluding status)
    display_columns = ["key", "summary"]
    
    # Combine all needed columns
    all_columns = model_columns + display_columns
    
    # Keep only needed columns and drop rows with missing values in model columns
    df_clean = df[all_columns].copy()
    df_clean = df_clean.dropna(subset=model_columns)
    
    return df_clean 