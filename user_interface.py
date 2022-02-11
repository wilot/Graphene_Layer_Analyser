"""User interface

Contains methods associated with handling the user interfaces."""

import os
import argparse
import sys
from typing import List, Tuple
import tkinter as tk
from tkinter.font import Font as tkFont

import tkinterdnd2
import matplotlib.pyplot as plt
import hyperspy.api as hs

import main
from diffraction_spots import SpotGroup
from pattern_processor import ProcessedResult


################
# Main methods #
################

def plot(filename: str, process_result: ProcessedResult, show_fig=False) -> plt.Figure:
    """Plots a figure showing the results of the analysis, returning the figure."""

    display_filename = filename.split("/")[-1]
    if not process_result.is_success():
        display_filename += " ERROR"

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
        print(f"Processing of {filename} complete.")


def save_figure(figure: plt.Figure, output_dir: str, source_filename: str):
    """Saves the figure."""

    _, source_filename = os.path.split(source_filename)
    save_filename = source_filename.split(".")[:-1]  # Strip filetype
    save_filename.append(".png")
    save_filename = ''.join(save_filename)
    output_filename = os.path.join(output_dir, save_filename)
    figure.savefig(output_filename)
    plt.close(figure)  # Clear up memory


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
    parser.add_argument("--filename", "-f", type=str, nargs="+", help=filename_help_text, required=True)
    parser.add_argument("--output", type=str, default="output", help=output_help_text)
    parser.add_argument("--show", "-s", action="store_true", help=show_help_text)
    parser.add_argument("--parallelize", "-p", action="store_true", help=parallel_help_text)

    args = parser.parse_args()

    if args.filename is None:
        parser.print_usage()
        sys.exit(0)

    valid_filenames = validate_filenames(args.filename)
    valid_output_directory = validate_output_directory(args.output)

    if not valid_filenames or not valid_output_directory:
        print("Failed to read filenames or output directory!")
        sys.exit(1)

    if args.show and args.parallelize:
        print("Cannot both parallelise and show.")
        sys.exit(1)

    return args.filename, args.output, args.show, args.parallelize


############################
# Graphical user interface #
############################

# TODO: Improve GUI

def spinup_gui():
    """Starts the GUI"""

    def main_handoff():
        """Gathers arguments from the GUI and hands-off control to main.py"""
        filenames = collect_files()
        valid_filenames = validate_filenames(filenames, "gui")
        output_directory = "output2/"  # TODO: Get from GUI
        valid_output_directory = validate_output_directory(output_directory, "gui")
        if not valid_filenames or not valid_output_directory:
            sys.exit(1)  # TODO: GUI-ify
        main.gui_invocation(filenames, output_directory)  # Handoff to main!

    # Window setup
    window = tkinterdnd2.TkinterDnD.Tk()
    window.title("Graphene Layer Analyser")
    width = 720
    height = 480
    screenwidth = window.winfo_screenwidth()
    screenheight = window.winfo_screenheight()
    alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    window.geometry(alignstr)
    window.resizable(width=False, height=False)

    # Title-label setup
    title_label = tk.Label(window)
    ft = tkFont(family='Times', size=22)
    title_label["font"] = ft
    title_label["fg"] = "#333333"
    title_label["justify"] = "center"
    title_label["text"] = "Graphene Layer Analyser"
    title_label.place(x=20,y=10,width=687,height=52)

    # Directions-label setup
    directions_label = tk.Label(window)
    ft = tkFont(family='Times', size=14)
    directions_label["font"] = ft
    directions_label["fg"] = "#333333"
    directions_label["justify"] = "center"
    directions_label["text"] = "Drag and drop files to be processed, then press begin."
    directions_label.place(x=20,y=70,width=349,height=38)

    # Filenames-listbox setup
    filenames_listbox = tk.Listbox(window)
    filenames_listbox["borderwidth"] = "1px"
    ft = tkFont(family='Times', size=12)
    filenames_listbox["font"] = ft
    filenames_listbox["fg"] = "#333333"
    filenames_listbox["justify"] = "center"
    filenames_listbox.place(x=340,y=70,width=369,height=388)

    # Make filenames-listbox drag&drop able
    filenames_listbox.drop_target_register(tkinterdnd2.DND_FILES)
    filenames_listbox.dnd_bind("<<Drop>>", lambda event: filenames_listbox.insert("end", event.data))

    def collect_files():
        """Records all the filenames from the listbox"""
        filenames = filenames_listbox.get(0, filenames_listbox.size()-1)  # The get method indices are inclusive...
        print(f"{filenames=}")
        print(f"type: {type(filenames)}")
        return filenames

    # Start-button setup
    start_button = tk.Button(window)
    start_button["bg"] = "#e9e9ed"
    ft = tkFont(family='Times', size=14)
    start_button["font"] = ft
    start_button["fg"] = "#000000"
    start_button["justify"] = "center"
    start_button["text"] = "Begin"
    start_button.place(x=20,y=130,width=298,height=35)
    start_button["command"] = main_handoff

    # Progress-message setup
    progress_message = tk.Message(window)
    ft = tkFont(family='Times', size=12)
    progress_message["font"] = ft
    progress_message["fg"] = "#333333"
    progress_message["justify"] = "center"
    progress_message["text"] = "Progress will be displayed here..."
    progress_message.place(x=20,y=180,width=261,height=275)

    window.mainloop()  # Blocking!


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

    if len(spot_groups) == 0:
        return

    group_color_labels = {'r': "Red", 'g': "Green", 'b': "Blue", 'c': "Cyan", 'm': "Magenta", 'y': "Yellow",
                          'w': "White"}
    group_colours = [group.colour for group in spot_groups]
    group_labels = list(map(lambda group: group_color_labels[group.colour], spot_groups))
    if len(spot_groups) < 5:  # If there's sufficient room in the figure
        group_labels = [label + ' ' + "group" for label in group_labels]

    bar_width = 0.35
    group_positions = [i + bar_width / 2 for i in range(len(spot_groups))]

    group_intensities = [group.group_intensity for group in spot_groups]
    group_intensities = [group_intensity / 1e3 for group_intensity in group_intensities]  # Plot kilo-counts
    group_uncertanties = [group.group_uncertainty / 1e3 for group in spot_groups]
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

    ax.bar(group_positions, group_intensities, yerr=group_uncertanties, color=group_colours, alpha=0.75, edgecolor='k')


##############
# Validation #
##############

def validate_filenames(filenames: List[str], mode="cli"):
    """Checks filenames for their files' existence, removing invalid filenames. Returns True if filenames are valid,
    and a list of the invalid filenames otherwise. """

    invalid_filename = lambda filename: not os.path.isfile(filename)
    invalid_filenames_list = list(filter(invalid_filename, filenames))
    if len(invalid_filenames_list) == 0:
        return True

    if mode == "gui":
        # TODO: Print debug to GUI
        pass
    else:
        print("The following filenames were not recognized.")
        print("\n".join(invalid_filenames_list))

    return False


def validate_output_directory(directory: str, mode="cli"):
    """Validates an output directory. Directory must exist and be empty"""

    if os.path.isdir(directory) and len(os.listdir(directory)) == 0:
        return True

    if mode == "gui":
        # TODO: Print debug to GUI
        pass
    else:
        print(f"The output directory '{directory}' was not found or was not empty.")

    return False