

import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

def _get_status_file_path(project_root: Path, config: Dict[str, Any]) -> Path:

    status_dir_relative = config['data']['status_dir'] 
    return project_root / status_dir_relative 

def get_last_processed_date(project_root: Path, config: Dict[str, Any], source_key: str) -> Optional[datetime.date]:

    status_file_path = _get_status_file_path(project_root, config)
    if not status_file_path.exists():
        return None
    
    try:
        with open(status_file_path, 'r') as f:
            status_data = yaml.safe_load(f)
            if status_data and 'last_processed_dates' in status_data and source_key in status_data['last_processed_dates']:
                date_str = status_data['last_processed_dates'][source_key]
                return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    except Exception as e:
        print(f"WARNING: Could not read status for '{source_key}' from {status_file_path}: {e}")
    return None

def update_last_processed_date(project_root: Path, config: Dict[str, Any], source_key: str, new_date: datetime.date):

    status_file_path = _get_status_file_path(project_root, config)
    status_file_path.parent.mkdir(parents=True, exist_ok=True) 

    status_data = {}
    if status_file_path.exists():
        try:
            with open(status_file_path, 'r') as f:
                status_data = yaml.safe_load(f) or {} 
        except Exception as e:
            print(f"WARNING: Error loading existing status file {status_file_path}: {e}. A new one will be created.")

    if 'last_processed_dates' not in status_data:
        status_data['last_processed_dates'] = {}
    
    status_data['last_processed_dates'][source_key] = new_date.strftime('%Y-%m-%d')

    try:
        with open(status_file_path, 'w') as f:
            yaml.safe_dump(status_data, f, default_flow_style=False)
        print(f"STATUS: Last processed date for '{source_key}' updated to {new_date} in {status_file_path}")
    except Exception as e:
        print(f"ERROR: Could not write status for '{source_key}' to {status_file_path}: {e}")

def set_current_execution_date(current_date: datetime.date):

    print(f"DEBUG: Pipeline execution date set to {current_date}")