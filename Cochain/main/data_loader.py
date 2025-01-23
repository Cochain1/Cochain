# data_loader.py

import json
import logging
from typing import Any, Dict, List
import yaml


def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load the configuration file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Dictionary of configuration parameters.
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Successfully loaded configuration file: {config_file}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: '{config_file}'. Please ensure the file exists at the specified path.")
        exit()
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file: {e}")
        exit()


def load_json_file(file_path: str) -> Any:
    """
    Generic JSON file loader.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Any: Loaded JSON data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded file: {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: '{file_path}'. Please ensure the file exists at the specified path.")
        exit()
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON file: {e}")
        exit()


def load_eval_data(file_path: str) -> List[Dict]:
    """
    Load evaluation data.

    Args:
        file_path (str): Path to the evaluation data file.

    Returns:
        List[Dict]: List of evaluation data.
    """
    return load_json_file(file_path)


def load_tree_data(file_path: str) -> Dict:
    """
    Load tree data.

    Args:
        file_path (str): Path to the tree data file.

    Returns:
        Dict: Dictionary of tree data.
    """
    return load_json_file(file_path)