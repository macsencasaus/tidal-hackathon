"""
Trains the U-net based on hdf_file data and config parameters
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import time

import numpy as np
import tensorflow as tf

from config_files import read_json_config
from data_processing import apply_sfft, get_normalized_data, get_time_info
from model2 import get_model

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


def get_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Trains and saves the U-net.")

    parser.add_argument(
        "--training-config",
        default="training.json",
        type=str,
        help="Name the JSON file the program uses to "
        "compile the model, Default: training.json",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="Choose to save the model after fitting, Default: True",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # PRELIMINARIES
    # -------------------------------------------------------------------------

    print("")
    print("TRAINING AND SAVING MODEL")
    print("")

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
    tf.random.set_seed(seed)

    # -------------------------------------------------------------------------
    # ACQUIRING THE TRAINING DATA
    # -------------------------------------------------------------------------

    hdf_file_name = config["training_hdf_file_path"]

    X_train, y_train = np.load("data/processed_data/sfft.npy")

    # -------------------------------------------------------------------------
    # ACQUIRING, COMPILING, AND TRAINING THE MODEL
    # -------------------------------------------------------------------------

    # Get time data
    time_info = get_time_info(hdf_file_name)
    sample_length = time_info["sample_length"]
    target_sampling_rate = time_info["target_sampling_rate"]
    img_size = int(sample_length * target_sampling_rate)

    # Load model
    print("Loading model... ", end="", flush=True)
    model = get_model()
    print("Done!", flush=True)

    # Compile model
    learning_rate = float(config["learning_rate"])
    loss = str(config["loss"])
    accuracy_metrics = list(config["accuracy_metrics"])
    print("Compiling model... ", end="", flush=True)
    model.compile(
        optimizer="adam",
        loss=loss,
        metrics=accuracy_metrics,
    )
    print("Done!", flush=True)

    # Fit model
    print("")
    print("FITTING MODEL: ", flush=True)
    print("")
    validation_split = float(config["validation_split"])
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
    )
    print("")

    # Save model
    if args.save_model:
        print("Saving model... ", end="", flush=True)
        model_name = str(config["model_name"])
        model_file_path = f"outputs/models/{model_name}"
        model.save(model_file_path)
        print("Done!", flush=True)

    # Print total runtime
    print("")
    print("Total runtime: {:.1f}s".format(time.time() - script_start))
    print("")
