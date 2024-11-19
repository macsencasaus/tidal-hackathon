"""
Processes data for 2D U-net applying short form fourier transform
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import time

import numpy as np

from config_files import read_json_config
from data_processing import apply_sfft, get_normalized_data

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


def get_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Trains and saves the U-net.")

    parser.add_argument(
        "--training-config",
        default="training2.json",
        type=str,
        help="Name the JSON file the program uses to "
        "compile the model, Default: training.json",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # PRELIMINARIES
    # -------------------------------------------------------------------------

    # Start stopwatch
    script_start = time.time()

    # Get command line arguments
    args = get_arguments()

    # Get JSON config file
    config_path = f"config_files/{args.training_config}"
    config = read_json_config(config_path)

    # Set seeds
    seed = config["random_seed"]
    np.random.seed(seed)

    # -------------------------------------------------------------------------
    # ACQUIRING THE TRAINING DATA
    # -------------------------------------------------------------------------

    hdf_file_name = config["training_hdf_file_path"]
    X_train, y_train = get_normalized_data(hdf_file_name)

    print("")
    print("Applying Short Form Fourier Transform...", end=" ", flush=True)
    X_train = apply_sfft(X_train)
    y_train = apply_sfft(y_train)
    print("Done!")
    print("")

    np.save("data/processed_data/sfft.npy", arr=[X_train, y_train])
