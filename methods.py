"""Utility methods."""

from typing import List

import numpy as np
import scipy.optimize
import scipy.ndimage
import matplotlib.pyplot as plt

from hyperspy.api import signals as hs_signals


def centre_of_mass(image: np.ndarray, lower_threshold_abs, show=False):
    """Determines the centre of mass of an image where values below a certain absolute threshold are ignored."""

    clipped_image = image.copy()
    clipped_image[clipped_image < lower_threshold_abs] = 0
    com = scipy.ndimage.center_of_mass(clipped_image)

    if show:
        fig, ax = plt.subplots()
        ax.imshow(clipped_image, vmin=0, vmax=200)
        com_patch = plt.Circle(com[::-1], 10, color='g', fill=True)
        ax.add_patch(com_patch)
        ax.set_title("Centre of mass clipped")
        plt.show()

    return com


def get_gaussian_fwhm(img, test_peak_coord):
    """Determines the gaussian FWHM of the peak at location specified by peak_coord."""

    gaussian_fit_sub_window_half_size = (100, 100)
    gaussian_fit_sub_window = img[test_peak_coord[0] - gaussian_fit_sub_window_half_size[0]:
                                  test_peak_coord[0] + gaussian_fit_sub_window_half_size[0],
                                  test_peak_coord[1] - gaussian_fit_sub_window_half_size[0]:
                                  test_peak_coord[1] + gaussian_fit_sub_window_half_size[0]]

    x = np.linspace(0, gaussian_fit_sub_window.shape[1], gaussian_fit_sub_window.shape[1])
    y = np.linspace(0, gaussian_fit_sub_window.shape[0], gaussian_fit_sub_window.shape[0])
    x, y = np.meshgrid(x, y)
    # Parameters: xpos, ypos, sigma, amp, baseline
    initial_guess = (gaussian_fit_sub_window.shape[1] / 2, gaussian_fit_sub_window.shape[0] / 2, 10, 1, 0)
    # subtract background and rescale image into [0,1], with floor clipping
    bg = np.percentile(gaussian_fit_sub_window, 5)
    img_scaled = np.clip((gaussian_fit_sub_window - bg) / (gaussian_fit_sub_window.max() - bg), 0, 1)
    popt, pcov = scipy.optimize.curve_fit(_gaussian,
                                          (x, y),
                                          img_scaled.ravel(),
                                          p0=initial_guess,
                                          bounds=((gaussian_fit_sub_window.shape[1] * 0.4,
                                                   gaussian_fit_sub_window.shape[0] * 0.4, 1, 0.5, -0.1),
                                                  (gaussian_fit_sub_window.shape[1] * 0.6,
                                                   gaussian_fit_sub_window.shape[0] * 0.6,
                                                   gaussian_fit_sub_window.shape[0] / 2, 1.5, 0.5)))
    xcenter, ycenter, sigma, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4]
    fwhm = np.abs(4 * sigma * np.sqrt(-0.5 * np.log(0.5)))
    return fwhm


def plot_segmented_image(peak_groups, centre, outer_background_radius,
                         signal: hs_signals.Signal2D, error=False):
    """Plots the hyperspy signal with circles showing the outer-background radii """

    fig, ax = plt.subplots()
    ax.imshow(signal.data, vmin=0, vmax=200)
    group_colors = 'r', 'g', 'b', 'c', 'm', 'y', 'w'
    for group_color, group in zip(group_colors, peak_groups):
        for peak_point in group.spots:
            circle = plt.Circle(peak_point[::-1], outer_background_radius, color=group_color, fill=False)
            ax.add_patch(circle)
    central_spot_patch = plt.Circle(centre[::-1], outer_background_radius / 2, color='g', fill=True)
    ax.add_patch(central_spot_patch)
    if error:
        ax.set_title("Error!")
    plt.show()


def plot_group_intensities(grouped_integrated_intensities):
    """Plots a bar chart of the integrated intensities of the respective groups."""

    fig, ax = plt.subplots()
    group_color_labels = "Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "White"
    group_labels = [group_color_labels[i]+' '+"group" for i in range(len(grouped_integrated_intensities))]
    width = 0.35
    group_positions = [i + width / 2 for i, _ in enumerate(group_labels)]
    ax.bar(group_positions, grouped_integrated_intensities)
    ax.set_ylabel("Summed Counts")
    ax.set_xticks(group_positions)
    ax.set_xticklabels(group_labels)
    plt.show()


def identify_thickness(grouped_integrated_intensities, current_filename):
    """Takes the an iterable of the integrated intensities of all the groups and determines the thickness."""

    if len(grouped_integrated_intensities) == 2:
        monolayer = grouped_integrated_intensities[0] * 0.9 < grouped_integrated_intensities[1] < \
                    grouped_integrated_intensities[1] * 1.1
        bilayer = grouped_integrated_intensities[0] * 0.4 < grouped_integrated_intensities[1] < \
                  grouped_integrated_intensities[1] * 0.6

        if bilayer:
            print("Bilayer or thicker positively ", end='')
        elif monolayer:
            print("Monolayer positively ", end='')
        else:
            print("Nothing positively ", end='')
        print("identified for file:\t", current_filename)


def get_cli_arguments() -> List[str]:
    """Handles the reading of command line arguments."""

    import argparse

    # TODO: Move these strings to a more sensible place.
    description = "A program to process a stack of diffraction patterns of Graphene and attempt to automatically " \
                  "determine if the sample is a monolayer or a few layers."
    filename_help_text = "A filename or list of filenames to be processed."

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("filename", type=str, nargs="+", help=filename_help_text)

    args = parser.parse_args()

    fetched_filenames = args.filename
    print(f"The filename arguments are:\t{fetched_filenames=}")
    print(fetched_filenames)

    # TODO: Perform filename validation

    if len(fetched_filenames) > 1:
        raise NotImplementedError("Multiple file processing is not implemented yet!")

    return fetched_filenames


def load_filenames():
    """Hardcoded loading of filenames on my pc. For DEBUGGING ONLY!!!"""

    import os

    data_directory = "data"
    data_filenames = []

    # Make a list of filenames
    for dataset_directory in os.listdir(data_directory):
        dataset_directory_path = os.path.join(data_directory, dataset_directory)
        data_filenames.append({
            "dataset_name": dataset_directory,
            "dataset_file_list": [
                os.path.join(dataset_directory_path, filename)
                for filename in os.listdir(dataset_directory_path)
                if filename[-4:] == ".dm3" or filename[-4:] == ".emd"
            ]
        })

    return data_filenames


def _gaussian(pos, xo, yo, sigma, amplitude, offset):
    """Returns a 2D gaussian function flattened to a 1D array."""

    g = offset + amplitude * np.exp(
        - (((pos[0] - xo) ** 2) / (2 * sigma ** 2) + ((pos[1] - yo) ** 2) / (2 * sigma ** 2))
    )

    return g.ravel()
