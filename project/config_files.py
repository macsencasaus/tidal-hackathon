"""
Provide functions for reading and parsing config files
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import json
import os

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


def read_json_config(file_path: str) -> dict:
    """
    Read in config JSON file for a few of the model's parameters

    Args:
        file_path: path of the config file

    Returns:
        A dictionary containing all the keys in the JSON file
    """

    # Make sure config file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Specified config file does not exist: {file_path}")

    # Open JSON file
    with open(file_path, "r") as json_file:
        config = json.load(json_file)

    return config
