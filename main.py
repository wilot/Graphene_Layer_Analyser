"""Test module

This module will primarily be used for ironing out the analytical techniques. Not to be used by users.
"""

from sys import exit

import numpy as np
import skimage.feature
import skimage.filters
import hyperspy.api as hs
import matplotlib.pyplot as plt

import diffraction_spots
import windows
import methods
import user_interface as ui


# TODO: Properly implement
# ui.get_cli_arguments()

# For debugging only, to be superseded by get_cli_arguments(). Modify with your own path to data while developing.
image_filenames = ui.load_filenames()
filename = image_filenames[0]["dataset_file_list"][18]
s = hs.load(filename)
print("Image loaded")

# First identify any peak, measure its Gaussian FWHM, filter the image according to this and re-detect peaks.
peaks_list = skimage.feature.peak_local_max(s.data, min_distance=100, threshold_rel=0.2, exclude_border=True,
                                            num_peaks=5)
peak_fwhm = np.median([methods.get_gaussian_fwhm(s.data, peak) for peak in peaks_list])
peak_fwhm = max(peak_fwhm, 5.)

# Tune-able parameters
selected_radius = peak_fwhm * 2
inner_background_radius = peak_fwhm * 4
outer_background_radius = peak_fwhm * 5
image_low_threshold = 100  # A threshold below which data is ignored in various methods
centre_within_radius = 200  # A radius from the centre of the image that the central spot can be assumed to be within
radial_group_threshold = max(0.2 / s.axes_manager[0].scale, 100)  # Radial segmentation threshold of 0.2/nm

# Primary peak-finding
dilated_image = skimage.filters.gaussian(s.data, sigma=peak_fwhm)
peaks_list = skimage.feature.peak_local_max(dilated_image, min_distance=int(outer_background_radius * 1.5),
                                            threshold_rel=0.05, exclude_border=True)
print(f"There are {len(peaks_list)} peaks initially identified")
if len(peaks_list) < 10:
    print("Less than 10 peaks found!")
    raise NotImplementedError

# Estimate the position of the zeroth-order diffraction spot by segmenting spots by radial distance from the image's
# centre-of-mass and then taking the median circumcentre of all permutations of spots within each group. Use this
# refined circumcentral zeroth-order spot position to refine radial segmentation.
central_peak_estimate = methods.centre_of_mass(s.data, lower_threshold_abs=image_low_threshold)
peak_groups = diffraction_spots.group_radially(central_peak_estimate, peaks_list, threshold=radial_group_threshold)
peak_groups = diffraction_spots.prune_spot_groups(peak_groups, min_spots=3)
circumcentres = []
for peak_group in peak_groups:
    try:
        group_circumcentres = peak_group.get_circumcentres(permitted_radius=centre_within_radius)
    except RuntimeError:  # Failed to find any circumcentres :(
        continue
    else:
        circumcentres.extend(group_circumcentres)
if len(circumcentres) > 0:
    central_spot = np.median(np.array(circumcentres), axis=0)
    peak_groups = diffraction_spots.group_radially(central_spot, peaks_list, threshold=radial_group_threshold)
else:
    central_spot = central_peak_estimate
    print("WARNING: Failed to best estimate the zeroth-order diffraction spot.")

# Now segment according to polar-angle too
peak_groups = diffraction_spots.group_azimuthally(central_spot, peak_groups, threshold=2)

peak_groups = diffraction_spots.prune_spot_groups(peak_groups)

grouped_integrated_intensities = []
image_window = windows.CircularWindow(selected_radius, inner_background_radius, outer_background_radius, s.data)
for group in peak_groups:
    group.calculate_integrated_intensity(image_window)

ui.assign_spot_group_colours(peak_groups)
ui.plot_segmented_image(peak_groups, central_spot, outer_background_radius, s)
ui.plot_group_intensities(peak_groups)

# DEBUGGING: for displaying the image window's masks on some random spot.
# fig, ax = plt.subplots()
# image_window.show_masks(*peak_groups[0].spots[1], ax)
# plt.show()
