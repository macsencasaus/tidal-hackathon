"""
Preprocesses .hdf files containing the waveforms and metadata.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import random as r

import h5py
import numpy as np
from numpy import abs, array, concatenate, max, newaxis, zeros

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


# def hdf_file_check(hdf_file_path: str):
#     # Check if the HDF file actually exists
#     if not os.path.exists(hdf_file_path):
#         raise FileNotFoundError(f'Hdf file "{hdf_file_path}" does not exist!')
#
#
# def get_normalized_data(hdf_file_path: str) -> tuple:
#     """
#     Pre-process and retrieve the input and label data
#
#     Args:
#         hdf_file_name: Path to the HDF file containg the data
#
#     Returns:
#         A tuple of the normalized inputs and labels
#     """
#
#     hdf_file_check(hdf_file_path)
#
#     with h5py.File(hdf_file_path, "r") as hdf_file:
#
#         hdf_dict: dict[str, h5py.Group | h5py.Dataset] = dict(hdf_file)
#
#         # Input data
#         inputs_h1 = array(hdf_dict["injection_samples"]["h1_strain"])[:, :, newaxis]
#         inputs_l1 = array(hdf_dict["injection_samples"]["l1_strain"])[:, :, newaxis]
#
#         # Label data
#         labels_h1 = array(hdf_dict["injection_parameters"]["h1_signal"])[:, :, newaxis]
#         labels_l1 = array(hdf_dict["injection_parameters"]["l1_signal"])[:, :, newaxis]
#
#         # Noise samples
#         noise_samples_h1 = array(hdf_dict["noise_samples"]["h1_strain"])[:, :, newaxis]
#         noise_samples_l1 = array(hdf_dict["noise_samples"]["l1_strain"])[:, :, newaxis]
#
#         # Injection snr
#         injection_snr = array(hdf_dict["injection_parameters"]["injection_snr"])
#
#     # Define normalizing functions
#     z_score = lambda x: (x - x.mean()) / x.std()
#     scaling = lambda x: x / max(abs(x))
#     multiply = lambda x, y: x * y
#
#     # Merging and normalizing input data
#     inputs = concatenate((inputs_h1, inputs_l1), axis=2)
#     inputs = array(list(map(z_score, inputs))) + 0.5
#
#     # Merging and normalizing label data
#     labels = concatenate((labels_h1, labels_l1), axis=2)
#     labels = map(scaling, labels)
#     labels = array(list(map(multiply, labels, injection_snr)))
#     labels = scaling(labels)
#     labels = (labels + 1) / 2
#
#     # Merging and normalizing noise samples
#     noise_samples = concatenate((noise_samples_h1, noise_samples_l1), axis=2)
#     noise_samples = array(list(map(z_score, noise_samples)))
#
#     # Concatenate noise samples to inputs
#     inputs = concatenate((inputs, noise_samples), axis=0)
#
#     # Concatenate noise sample labels to labels
#     labels = concatenate((labels, zeros(noise_samples.shape) + 0.5), axis=0)
#
#     return inputs, labels


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

IMG_SIZE = 16384


def hdf_file_check(hdf_file_path: str):
    # Check if the HDF file actually exists
    if not os.path.exists(hdf_file_path):
        raise FileNotFoundError(f'Hdf file "{hdf_file_path}" does not exist!')


def get_normalized_data(hdf_file_path: str) -> tuple:
    """
    Pre-process and retrieve the input and label data

    Args:
        hdf_file_name: Path to the HDF file containg the data

    Returns:
        A tuple of the normalized inputs and labels
    """

    hdf_file_check(hdf_file_path)

    with h5py.File(hdf_file_path, "r") as hdf_file:

        hdf_dict: dict[str, h5py.Group | h5py.Dataset] = dict(hdf_file)

        injection_samples = hdf_dict["injection_samples"]
        injection_parameters = hdf_dict["injection_parameters"]
        noise_samples = hdf_dict["noise_samples"]

        n_injections = len(injection_samples["h1_strain"])
        n_noise_samples = len(noise_samples["h1_strain"])

        h1_cropped_injection_samples = np.zeros((n_injections, IMG_SIZE))
        l1_cropped_injection_samples = np.zeros((n_injections, IMG_SIZE))

        h1_cropped_injection_signal = np.zeros((n_injections, IMG_SIZE))
        l1_cropped_injection_signal = np.zeros((n_injections, IMG_SIZE))

        h1_cropped_noise_samples = np.zeros((n_noise_samples, IMG_SIZE))
        l1_cropped_noise_samples = np.zeros((n_noise_samples, IMG_SIZE))

        for i in range(n_injections):
            start_time = r.randint(200, 14000)
            end_time = start_time + IMG_SIZE

            h1_cropped_injection_samples[i] = injection_samples["h1_strain"][i][
                start_time:end_time
            ]
            l1_cropped_injection_samples[i] = injection_samples["l1_strain"][i][
                start_time:end_time
            ]

            h1_cropped_injection_signal[i] = injection_parameters["h1_signal"][i][
                start_time:end_time
            ]
            l1_cropped_injection_signal[i] = injection_parameters["l1_signal"][i][
                start_time:end_time
            ]

        for i in range(n_noise_samples):
            h1_cropped_noise_samples[i] = noise_samples["h1_strain"][i][:IMG_SIZE]
            l1_cropped_noise_samples[i] = noise_samples["l1_strain"][i][:IMG_SIZE]

        X_train_h1 = np.concatenate(
            (h1_cropped_injection_samples, h1_cropped_noise_samples)
        ).reshape((n_injections + n_noise_samples, IMG_SIZE, -1))
        X_train_l1 = np.concatenate(
            (l1_cropped_injection_samples, l1_cropped_noise_samples)
        ).reshape((n_injections + n_noise_samples, IMG_SIZE, -1))

        Y_train_h1 = np.concatenate(
            (h1_cropped_injection_signal, np.zeros((n_noise_samples, IMG_SIZE)))
        ).reshape((n_injections + n_noise_samples, IMG_SIZE, -1))
        Y_train_l1 = np.concatenate(
            (l1_cropped_injection_signal, np.zeros((n_noise_samples, IMG_SIZE)))
        ).reshape((n_injections + n_noise_samples, IMG_SIZE, -1))

        X_train = np.concatenate((X_train_h1, X_train_l1), axis=2)
        Y_train = np.concatenate((Y_train_h1, Y_train_l1), axis=2)

        for i, a in enumerate(Y_train):
            Y_train[i] = a / np.max(np.abs(1 if np.max(a) == 0 else a))

        for i, a in enumerate(X_train):
            X_train[i] = (a - a.mean()) / (1 if a.std() == 0 else a.std()) + 0.5

        for i in range(n_injections):
            Y_train[i,] *= injection_parameters["injection_snr"][i]

        Y_train /= np.max(np.abs(Y_train))

        Y_train = (Y_train + 1) / 2

    return X_train, Y_train


def get_raw_data(hdf_file_path: str) -> tuple:
    """
    Retrieves the raw data for plotting

    Args:
        hdf_file_name: name of the HDF file used for testing

    Returns:
        A tuple of the input data and the labels for plotting
    """

    hdf_file_check(hdf_file_path)

    with h5py.File(hdf_file_path, "r") as hdf_file:

        hdf_dict: dict[str, h5py.Group | h5py.Dataset] = dict(hdf_file)

        # Input data
        inputs_h1 = array(hdf_dict["injection_samples"]["h1_strain"])[:, :, newaxis]
        inputs_l1 = array(hdf_dict["injection_samples"]["l1_strain"])[:, :, newaxis]

        # Label data
        labels_h1 = array(hdf_dict["injection_parameters"]["h1_signal"])[:, :, newaxis]
        labels_l1 = array(hdf_dict["injection_parameters"]["l1_signal"])[:, :, newaxis]

        # Noise samples
        noise_samples_h1 = array(hdf_dict["noise_samples"]["h1_strain"])[:, :, newaxis]
        noise_samples_l1 = array(hdf_dict["noise_samples"]["l1_strain"])[:, :, newaxis]

    # Merging and normalizing input data
    inputs = concatenate((inputs_h1, inputs_l1), axis=2)

    # Merging and normalizing label data
    labels = concatenate((labels_h1, labels_l1), axis=2)
    for idx, label in enumerate(labels):
        labels[idx] /= np.max(np.abs(label))

    # Merging and normalizing noise samples
    noise_samples = concatenate((noise_samples_h1, noise_samples_l1), axis=2)

    # Concatenate noise samples to inputs
    inputs = concatenate((inputs, noise_samples), axis=0)

    # Concatenate noise sample labels to labels
    labels = concatenate((labels, zeros(noise_samples.shape)), axis=0)

    return inputs, labels


def get_injection_parameters(hdf_file_path: str) -> dict:
    """
        Retrieves the injection parametdictionary = dict(a=[1,2,3], b=[True,False,True])

    array_size = len(next(iter(dictionary.values())))

    print(array_size)ers of HDF file

        Returns:
            Numpy array of dictionaries of the injection parameters

           Numpy array of dictionaries of the injection parameters
    """

    hdf_file_check(hdf_file_path)

    with h5py.File(hdf_file_path, "r") as hdf_file:

        hdf_dict: dict[str, h5py.Group | h5py.Dataset] = dict(hdf_file)

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
        injection_parameters_dict = {}
        for key in keys:
            injection_parameters_dict[key] = array(
                hdf_dict["injection_parameters"][key]
            )

    return injection_parameters_dict


def get_time_info(hdf_file_path: str) -> dict:
    """
    Retrieves the event time of an event of a given sample

    Args:
        hdf_file_path: path to the hdf file with data

    Returns:
        A float of the event time
    """

    hdf_file_check(hdf_file_path)

    with h5py.File(hdf_file_path, "r") as hdf_file:

        hdf_dict: dict[str, h5py.Group | h5py.Dataset] = dict(hdf_file)

        seconds_before_event = float(
            hdf_dict["static_arguments"].attrs["seconds_before_event"]
        )
        seconds_after_event = float(
            hdf_dict["static_arguments"].attrs["seconds_after_event"]
        )
        sample_length = float(hdf_file["static_arguments"].attrs["sample_length"])
        target_sampling_rate = float(
            hdf_dict["static_arguments"].attrs["target_sampling_rate"]
        )

    return dict(
        seconds_before_event=seconds_before_event,
        seconds_after_event=seconds_after_event,
        sample_length=sample_length,
        target_sampling_rate=target_sampling_rate,
    )


from scipy.signal import butter, filtfilt, stft


def apply_sfft(signals: np.ndarray, nperseg=128, nfft=512, fs=2048) -> np.ndarray:

    def highpass_filter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="high", analog=False)
        return filtfilt(b, a, data)

    n = len(signals)

    output = np.zeros((n, 256, 256, 2), dtype=complex)

    for i in range(n):
        for ch in range(2):
            filtered_signal = highpass_filter(signals[i, :, ch], 10, fs)
            _, _, Zxx = stft(
                filtered_signal,
                fs=fs,
                nperseg=nperseg,
                nfft=nfft,
            )
            output[i, :, :, ch] = np.log1p(np.abs(Zxx[:256, :256]))

    return output
