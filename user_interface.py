"""User interface

Contains methods associated with handling the user interfaces."""

import os
import argparse
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import hyperspy.api as hs

from diffraction_spots import SpotGroup
from pattern_processor import ProcessedResult


################
# Main methods #
################

def plot(filename: str, process_result: ProcessedResult, show_fig=False) -> plt.Figure:
    """Plots a figure showing the results of the analysis, returning the figure."""

    display_filename = filename.split("/")[-1]

    assign_spot_group_colours(process_result.spot_groups)

    fig, (segmentation_axis, intensities_axis) = plt.subplots(nrows=1, ncols=2, dpi=280, figsize=(10, 4.75))
    plot_segmented_image(segmentation_axis, process_result.spot_groups, process_result.central_spot_coordinate,
                         process_result.outer_background_radius, process_result.diffraction_pattern)
    plot_group_intensities(intensities_axis, process_result.spot_groups)
    fig.suptitle(display_filename)

    if show_fig:
        plt.show()

    return fig


def load_signals(filenames: List[str]):
    """Generator that takes a list of filenames and yields their loaded Hyperspy Signal2D objects."""

    for filename in filenames:
        try:
            signal = hs.load(filename)
        except Exception as error:
            print(f"There was a error processing {filename}.")
            print(error)
            continue
        yield signal


def save_figure(figure: plt.Figure, output_dir: str, source_filename: str):
    """Saves the figure."""

    _, source_filename = os.path.split(source_filename)
    save_filename = source_filename.split(".")[:-1]  # Strip filetype
    save_filename.append(".png")
    save_filename = ''.join(save_filename)
    output_filename = os.path.join(output_dir, save_filename)
    figure.savefig(output_filename)


##########################
# Command line interface #
##########################

def get_cli_arguments() -> Tuple[List[str], str, bool, bool]:
    """Handles the reading of command line arguments. Returns a list of the filenames to be processed, the output
    directory and whether the figures should be shown or saved."""

    description = "A program to process a stack of diffraction patterns of Graphene and attempt to automatically " \
                  "determine if the sample is a monolayer or a few layers."
    filename_help_text = "A filename or list of filenames to be processed."
    output_help_text = "Directory where output files should be stored. This must already exist at the location " \
                       "specified and be empty. Default is 'output/'."
    show_help_text = "Display figures rather than save them."
    parallel_help_text = "Parallelise the processing of files. Note, cannot be used with '--show'."

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--filename", "-f", type=str, nargs="+", help=filename_help_text)
    parser.add_argument("--output", type=str, default="output", help=output_help_text)
    parser.add_argument("--show", "-s", action="store_true", help=show_help_text)
    parser.add_argument("--parallelize", "-p", action="store_true", help=parallel_help_text)

    args = parser.parse_args()

    if args.filename is None:
        # Start GUI
        raise NotImplementedError

    validate_filenames(args.filename)
    validate_output_directory(args.output)
    if args.show and args.parallelize:
        print("Cannot both parallelise and show.")
        sys.exit(1)

    return args.filename, args.output, args.show, args.parallelize


############################
# Graphical user interface #
############################

# TODO: GUI

####################
# Plotting methods #
####################

def assign_spot_group_colours(spot_groups: List[SpotGroup]):
    group_colors = 'r', 'g', 'b', 'c', 'm', 'y', 'w'
    for group_index, spot_group in enumerate(spot_groups):
        colour_index = group_index % len(group_colors)
        spot_group.colour = group_colors[colour_index]


def plot_segmented_image(ax: plt.Axes, peak_groups, centre, circle_radius, signal):
    """Plots the Hyperspy signal with circles showing the outer-background radii."""

    ax.imshow(signal.data, vmin=0, vmax=200)
    for group_number, group in enumerate(peak_groups):
        for peak_point in group.spots:
            circle = plt.Circle(peak_point[::-1], circle_radius, color=group.colour, fill=False)
            ax.add_patch(circle)
    central_spot_patch = plt.Circle(centre[::-1], circle_radius / 2, color='w', fill=True)
    ax.add_patch(central_spot_patch)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def plot_group_intensities(ax: plt.Axes, spot_groups: List[SpotGroup]):
    """Plots a bar chart of the integrated intensities of the respective groups."""

    group_color_labels = {'r': "Red", 'g': "Green", 'b': "Blue", 'c': "Cyan", 'm': "Magenta", 'y': "Yellow",
                          'w': "White"}
    group_colours = [group.colour for group in spot_groups]
    group_labels = [group_color_labels[group.colour] + ' ' + "group" for group in spot_groups]

    bar_width = 0.35
    group_positions = [i + bar_width / 2 for i in range(len(spot_groups))]

    group_intensities = [group.group_intensity for group in spot_groups]
    group_intensities = [group_intensity / 1e3 for group_intensity in group_intensities]  # Plot kilo-counts
    group_uncertanties = [group.group_uncertainty for group in spot_groups]
    group_uncertanties = [uncert / 1e3 if uncert is not None else 0. for uncert in group_uncertanties]  # Strip Nones
    group_uncertanties_text = ["{:.2%}".format(uncertainty / intensity) for uncertainty, intensity in
                               zip(group_uncertanties, group_intensities)]  # Pretty string format
    # Plot as kilo counts instead
    text_y_offset = max(group_intensities) * 0.02  # Height above the bar to add the text displaying uncertainty

    ax.yaxis.set_major_formatter("{x:.0f}")
    ax.set_ylabel("Summed Counts / 1E3")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.set_xticks(group_positions)
    ax.set_xticklabels(group_labels)

    for x_pos, intensity, text in zip(group_positions, group_intensities, group_uncertanties_text):
        y_pos = intensity + text_y_offset
        ax.text(x_pos, y_pos, text, fontsize="small")

    ax.bar(group_positions, group_intensities, yerr=group_uncertanties, color=group_colours, alpha=0.5)


##############
# Validation #
##############

def validate_filenames(filenames: List[str]):
    """Checks filenames for their files' existence, removing invalid filenames. Returns True if filenames are valid,
    and a list of the invalid filenames otherwise. """

    invalid_filename = lambda filename: not os.path.isfile(filename)
    invalid_filenames_list = list(filter(invalid_filename, filenames))
    if len(invalid_filenames_list) == 0:
        return
    print("The following filenames were not recognized.")
    print("\n".join(invalid_filenames_list))
    sys.exit(1)


def validate_output_directory(directory: str):
    """Validates an output directory. Directory must exist and be empty"""

    if os.path.isdir(directory) and len(os.listdir(directory)) == 0:
        return
    print(f"The output directory '{directory}' was not found or was not empty.")
    sys.exit(1)
