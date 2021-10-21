"""User interface

Contains methods associated with handling the user interfaces."""

import os.path
import argparse
from typing import List

import matplotlib.pyplot as plt

from diffraction_spots import SpotGroup


def assign_spot_group_colours(spot_groups: List[SpotGroup]):
    group_colors = 'r', 'g', 'b', 'c', 'm', 'y', 'w'
    for group_index, spot_group in enumerate(spot_groups):
        colour_index = group_index % len(group_colors)
        spot_group.colour = group_colors[colour_index]


def plot_segmented_image(peak_groups, centre, outer_background_radius,
                         signal, error=False):
    """Plots the hyperspy signal with circles showing the outer-background radii."""

    fig, ax = plt.subplots()
    ax.imshow(signal.data, vmin=0, vmax=200)
    for group_number, group in enumerate(peak_groups):
        for peak_point in group.spots:
            circle = plt.Circle(peak_point[::-1], outer_background_radius, color=group.colour, fill=False)
            ax.add_patch(circle)
    central_spot_patch = plt.Circle(centre[::-1], outer_background_radius / 2, color='w', fill=True)
    ax.add_patch(central_spot_patch)
    if error:
        ax.set_title("Error!")
    plt.show()


def plot_group_intensities(spot_groups: List[SpotGroup]):
    """Plots a bar chart of the integrated intensities of the respective groups."""

    fig, ax = plt.subplots()
    group_color_labels = {'r': "Red", 'g': "Green", 'b': "Blue", 'c': "Cyan", 'm': "Magenta", 'y': "Yellow",
                          'w': "White"}
    group_colours = [group.colour for group in spot_groups]
    group_labels = [group_color_labels[group.colour] + ' ' + "group" for group in spot_groups]
    width = 0.35
    group_positions = [i + width / 2 for i in range(len(spot_groups))]
    group_intensities = [group.group_intensity for group in spot_groups]
    group_uncertanties = [group.group_uncertainty for group in spot_groups]
    group_uncertanties = [uncert if uncert is not None else 0 for uncert in group_uncertanties]
    print("Raw uncertainties:", group_uncertanties)
    print("Relative uncertainties:", [str((uncert / intens)*100)+'%' for uncert, intens in zip(group_uncertanties, group_intensities)])
    ax.bar(group_positions, group_intensities, yerr=group_uncertanties, color=group_colours, alpha=0.5)
    ax.set_ylabel("Summed Counts")
    ax.set_xticks(group_positions)
    ax.set_xticklabels(group_labels)
    plt.show()


def get_cli_arguments() -> List[str]:
    """Handles the reading of command line arguments."""

    description = "A program to process a stack of diffraction patterns of Graphene and attempt to automatically " \
                  "determine if the sample is a monolayer or a few layers."
    filename_help_text = "A filename or list of filenames to be processed."

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("filename", type=str, nargs="+", help=filename_help_text)

    args = parser.parse_args()

    fetched_filenames = args.filename
    # print(f"The filename arguments are:\t{fetched_filenames=}")

    valid_filenames = validate_filenames(fetched_filenames)

    if len(fetched_filenames) > 1:
        raise NotImplementedError("Multiple file processing is not implemented yet!")

    return fetched_filenames


def validate_filenames(filenames: List[str]):
    """Checks filenames for their files' existence, removing invalid filenames."""

    is_valid_filename = lambda filename: os.path.isfile(filename)
    return filter(is_valid_filename, filenames)



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
