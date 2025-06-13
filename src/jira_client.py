import os
import requests
import pandas as pd
from requests.auth import HTTPBasicAuth
import time
from typing import Optional, List, Dict, Any

class JiraClient:
    """
    A class that helps us talk to Jira and get sprint data.
    It handles things like logging in and getting information about tasks.
    """
    def __init__(self, domain=None, email=None, api_token=None, project_key=None, board_id=None):
        # Get login information from environment variables or parameters
        self.domain = domain or os.getenv("JIRA_DOMAIN")
        self.email = email or os.getenv("JIRA_EMAIL")
        self.api_token = api_token or os.getenv("JIRA_API_TOKEN")
        self.project_key = project_key or os.getenv("JIRA_PROJECT_KEY")
        self.board_id = board_id or os.getenv("JIRA_BOARD_ID")
        
        # Set up login credentials
        self.auth = HTTPBasicAuth(self.email, self.api_token)
        self.headers = {"Accept": "application/json"}
        
        # Make sure we don't send too many requests too quickly
        self.last_request_time = 0
        self.min_request_interval = 0.5  # wait 0.5 seconds between requests
        
        # Check if we have all the required login information
        self._validate_config()
    
    def _validate_config(self):
        """Check if we have all the required login information"""
        if not all([self.domain, self.email, self.api_token]):
            raise ValueError("Missing required Jira configuration. Please set JIRA_DOMAIN, JIRA_EMAIL, and JIRA_API_TOKEN.")
        
        if not self.domain.endswith('.atlassian.net'):
            raise ValueError("Invalid Jira domain. Must end with .atlassian.net")
    
    def _rate_limit(self):
        """Wait a bit between requests to be nice to Jira's servers"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()
    
    def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Send a request to Jira and handle any errors"""
        self._rate_limit()
        try:
            response = requests.request(method, url, headers=self.headers, auth=self.auth, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if response.status_code == 401:
                raise ValueError("Authentication failed. Please check your Jira credentials.")
            elif response.status_code == 403:
                raise ValueError("Access denied. Please check your Jira permissions.")
            elif response.status_code == 429:
                raise ValueError("Rate limit exceeded. Please try again later.")
            else:
                raise ValueError(f"Jira API request failed: {str(e)}")

    def get_boards(self) -> List[Dict[str, Any]]:
        """
        Get all scrum boards from Jira.
        If a project key is set, only get boards for that project.
        """
        url = f"https://{self.domain}/rest/agile/1.0/board"
        boards = []
        start_at = 0
        
        # Jira gives us data in pages, so we need to get all pages
        while True:
            data = self._make_request('GET', url, params={
                "maxResults": 50,
                "startAt": start_at
            })
            
            boards.extend(data.get("values", []))
            if data.get("isLast", True) or len(data.get("values", [])) == 0:
                break
            start_at += len(data.get("values", []))
        
        # Only keep scrum boards
        scrum_boards = [b for b in boards if b.get("type") == "scrum"]
        if self.project_key:
            scrum_boards = [b for b in scrum_boards if b.get("location", {}).get("projectKey") == self.project_key]
        return scrum_boards

    def get_sprints(self, board_id: Optional[str] = None, count: int = 30) -> List[Dict[str, Any]]:
        """Get the most recent finished sprints for a board"""
        board_id = board_id or self.board_id
        url = f"https://{self.domain}/rest/agile/1.0/board/{board_id}/sprint"
        
        data = self._make_request('GET', url, params={
            "state": "closed",
            "maxResults": 100,
            "startAt": 0
        })
        
        all_sprints = data.get("values", [])
        if len(all_sprints) < count:
            print(f"⚠️ Warning: Only {len(all_sprints)} sprints found, requested {count}.")
            return all_sprints
        return all_sprints[-count:]

    def get_open_sprints(self, board_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all current and future sprints for a board"""
        board_id = board_id or self.board_id
        url = f"https://{self.domain}/rest/agile/1.0/board/{board_id}/sprint"
        
        data = self._make_request('GET', url, params={
            "state": "active,future",
            "maxResults": 100,
            "startAt": 0
        })
        
        return data.get("values", [])

    def get_issues_for_sprint(self, sprint_id: str) -> List[Dict[str, Any]]:
        """Get all tasks in a sprint"""
        url = f"https://{self.domain}/rest/agile/1.0/sprint/{sprint_id}/issue"
        
        data = self._make_request('GET', url, params={"maxResults": 100})
        return data.get("issues", [])

    def parse_issue(self, issue: Dict[str, Any], sprint_end: Optional[pd.Timestamp]) -> Dict[str, Any]:
        """
        Convert a Jira task into a format our machine learning model can use.
        Gets information like when it was created, when it was finished, and its status.
        """
        try:
            fields = issue["fields"]
            created = pd.to_datetime(fields["created"])
            resolutiondate = pd.to_datetime(fields.get("resolutiondate")) if fields.get("resolutiondate") else None
            time_spent = fields.get("timespent")
            original_estimate = fields.get("timeoriginalestimate")
            
            # Check if the task was completed
            status_name = fields["status"]["name"].lower()
            is_closed = status_name in ["done", "closed", "resolved"]
            
            # Create a dictionary with all the information we need
            return {
                "key": issue["key"],
                "summary": fields.get("summary", ""),
                "original_estimate": original_estimate if original_estimate else None,
                "assignee": fields["assignee"]["displayName"] if fields["assignee"] else None,
                "issue_type": fields["issuetype"]["name"],
                "comment_count": fields["comment"]["total"],
                "created": created,
                "resolved": resolutiondate,
                "time_spent": time_spent if time_spent else None,
                "sprint_success": int(is_closed and (resolutiondate is None or resolutiondate <= sprint_end)),
                "days_in_sprint": (sprint_end - created).days if sprint_end and created else None
            }
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid issue data format: {str(e)}")

