"""
Plots the input, label, and prediction of a certain sample in the testing sample
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from config_files import read_json_config
from data_processing import get_injection_parameters, get_raw_data, get_time_info

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


def get_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Plots the input, label, and prediction of one sample "
        "from both detectors"
    )

    parser.add_argument(
        "--testing-config",
        default="test.json",
        type=str,
        help="Name of the JSON file the model used to test, Default: testing.json",
    )
    parser.add_argument(
        "--sample-id",
        help="ID of the sample to be view (an integer "
        "between 0 and n_injection_samples + n_noise_samples),"
        "Default: 0",
        type=int,
        default=0,
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":

    # Start stopwatch
    script_start = time.time()

    # Get command line arguments
    args = get_arguments()

    # Get JSON config file
    config_path = f"config_files/{args.testing_config}"
    config = read_json_config(config_path)

    # Get input data & labels
    hdf_file_name = config["testing_hdf_file_name"]
    inputs, labels = get_raw_data(hdf_file_name)

    # Get injection parameters
    injection_parameters = get_injection_parameters(hdf_file_name)

    # Get snrs
    snrs = injection_parameters["injection_snr"]

    # Check if sample id is in bounds
    sample_id = int(args.sample_id)
    if args.sample_id < 0 or args.sample_id >= len(inputs):
        raise IndexError("Sample-id is not within bounds")

    # Check if the sample has injection
    has_injection = sample_id < len(snrs)

    # Get predictions
    model_path = config["model_path"]

    output_name = config["output_name"]

    preds_name = f"{output_name}.npy"
    preds_path = f"outputs/predictions/{preds_name}"
    preds = np.load(preds_path)

    # Unnormalize predictions
    preds = preds * 2 - 1
    preds = np.array(list(map(lambda x: x / np.max(np.abs(x)), preds)))

    # Get event time
    time_info_dict = get_time_info(hdf_file_name)
    seconds_before_event = time_info_dict["seconds_before_event"]
    seconds_after_event = time_info_dict["seconds_after_event"]
    target_sampling_rate = time_info_dict["target_sampling_rate"]
    sample_length = time_info_dict["sample_length"]

    if has_injection:
        time_range = np.arange(
            -seconds_before_event, seconds_after_event, 1 / target_sampling_rate
        )
    else:
        time_range = np.arange(0, sample_length, 1 / target_sampling_rate)

    # Create figure
    fig, ax1 = plt.subplots(nrows=3, ncols=2)

    ax1[0][0].set_ylabel("Whitened Signal Strain", color="C0", fontsize=8)
    ax1[1][0].set_ylabel("Raw Signal Strain", color="C1", fontsize=8)
    ax1[2][0].set_ylabel("U-Net Prediction Strain", color="C2", fontsize=8)
    ax1[0][1].set_yticklabels([])
    ax1[1][1].set_yticklabels([])
    ax1[2][1].set_yticklabels([])
    ax1[0][1].tick_params(left=False)
    ax1[1][1].tick_params(left=False)
    ax1[2][1].tick_params(left=False)

    for idx, detectors in enumerate(["H1", "L1"]):

        ax1[0][idx].set_title(detectors)

        ax1[0][idx].plot(time_range, inputs[sample_id, :, idx])

        ax1[0][idx].set_ylim(-120, 120)
        ax1[0][idx].tick_params("y", colors="C0", labelsize=8)

        ax1[1][idx].set_ylim(-1.2, 1.2)

        if has_injection:
            ax1[1][idx].plot(time_range, labels[sample_id, :, idx], color="C1")
        else:
            ax1[1][idx].plot(
                time_range,
                np.zeros(int(sample_length * target_sampling_rate)),
                color="C1",
            )

        ax1[1][idx].tick_params("y", colors="C1", labelsize=8)

        ax1[2][idx].plot(time_range, preds[sample_id, :, idx], color="C2")

        ax1[2][idx].set_ylim(-1.2, 1.2)
        ax1[2][idx].tick_params("y", colors="C2", labelsize=8)

        # Focuses the view around the event if sample has an injection
        if has_injection:
            ax1[0][idx].set_xlim(-0.15, 0.05)
            ax1[1][idx].set_xlim(-0.15, 0.05)
            ax1[2][idx].set_xlim(-0.15, 0.05)
            ax1[0][idx].axvline(x=0, color="black", ls="--", lw=1)
            ax1[1][idx].axvline(x=0, color="black", ls="--", lw=1)
            ax1[2][idx].axvline(x=0, color="black", ls="--", lw=1)
        else:
            ax1[0][idx].set_xlim(0, sample_length)
            ax1[1][idx].set_xlim(0, sample_length)
            ax1[2][idx].set_xlim(0, sample_length)

        # Set x-labels
        ax1[0][idx].set_xticklabels([])
        ax1[1][idx].set_xticklabels([])
        ax1[2][idx].set_xlabel("Time from event (sec)")

    # Adds subtitle for injection parameters
    if has_injection:
        keys = (
            "mass1",
            "mass2",
            "spin1z",
            "spin2z",
            "ra",
            "dec",
            "coa_phase",
            "inclination",
            "polarization",
            "injection_snr",
        )
        string = ", ".join(
            ["{} = {:.2f}".format(_, injection_parameters[_][sample_id]) for _ in keys]
        )
    else:
        string = "(sample does not contain an injection)"

    plt.figtext(0.5, 0.9, f"Injection Parameters:\n{string}", fontsize=8, ha="center")

    # Adjust the size and spacing of the subplots
    plt.gcf().set_size_inches(12, 6, forward=True)
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.subplots_adjust(wspace=0.05, hspace=0)

    plt.suptitle(f"Sample #{sample_id}", y=0.975)

    # Save the plot at the given location
    print("Saving plot... ", end="", flush=True)
    try:
        plt.savefig(
            f"outputs/figures/injection_{sample_id}.png",
            bbox_inches="tight",
            pad_inches=0.3,
        )
    except FileNotFoundError:
        os.mkdir("outputs/figures")
        plt.savefig(
            f"outputs/figures/injection_{sample_id}.png",
            bbox_inches="tight",
            pad_inches=0.3,
        )
    print("Done!", flush=True)
