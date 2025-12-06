import json
import os

def load_config(config_path="config.json"):
    """
    Loads the configuration from a JSON file.
    
    Args:
        config_path (str): Path to the config.json file.
        
    Returns:
        dict: The configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
