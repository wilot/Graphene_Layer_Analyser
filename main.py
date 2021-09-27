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


methods.get_cli_arguments()

image_filenames = methods.load_filenames()
filename = image_filenames[0]["dataset_file_list"][8]
s = hs.load(filename)
print("Image loaded")

print("Initial peak finding")
# First identify any peak, measure its Gaussian FWHM, filter the image according to this and re-detect peaks.
peaks_list = skimage.feature.peak_local_max(s.data, min_distance=100, threshold_rel=0.2, exclude_border=True,
                                            num_peaks=5)
print("Getting gaussian fwhm")
peak_fwhm = np.median([methods.get_gaussian_fwhm(s.data, peak) for peak in peaks_list])
peak_fwhm = max(peak_fwhm, 5.)
print("Second peak finding")
peaks_list = skimage.feature.peak_local_max(s.data, min_distance=int(peak_fwhm * 10), threshold_rel=0.2,
                                            exclude_border=True)
print(f"Image filtered, {len(peaks_list)} peaks identified")
if len(peaks_list) < 12:
    print("Less than 12 peaks found, quitting!")
    raise NotImplementedError

selected_radius = peak_fwhm * 2
inner_background_radius = peak_fwhm * 4
outer_background_radius = peak_fwhm * 5
image_low_threshold = 100  # A threshold below which data is ignored in various methods
centre_within_radius = 200  # A radius from the centre of the image that the central spot can be assumed to be within
radial_group_threshold = max(0.2 / s.axes_manager[0].scale, 100)  # Set a threshold of 0.2/nm

# Now refine centre
central_peak_estimate = methods.centre_of_mass(s.data, lower_threshold_abs=image_low_threshold)
peak_groups = diffraction_spots.group_radially(central_peak_estimate, peaks_list, threshold=radial_group_threshold)
print("Initially detected", len(peak_groups), "groups.")
# Determine the median circumcentre of all combinations of triplets of spots in each group
median_grouped_circumcentres = []
for peak_group in peak_groups:
    group_circumcentre = peak_group.get_median_circumcentre(permitted_radius=centre_within_radius)
    if group_circumcentre is not None:
        median_grouped_circumcentres.append(group_circumcentre)
print(len(median_grouped_circumcentres), " median grouped circumcentres found.")
if len(median_grouped_circumcentres) > 0:
    median_circumcentre = np.median(np.array(median_grouped_circumcentres), axis=0)
    print("Refining radial groups")
    peak_groups = diffraction_spots.group_radially(median_circumcentre, peaks_list, threshold=radial_group_threshold)
    print("Radial groups refined")
else:
    # Then peak refinement failed...
    median_circumcentre = central_peak_estimate
    print("Radial group refinement failed")

print("Direct-beam spot position identified.")

# The beam-stop will not block more than 2 spots in any single group.
peak_groups = list(filter(lambda peak_group: len(peak_group) >= 4, peak_groups))
print("There are now", len(peak_groups), "radial groups.")

grouped_integrated_intensities = []
image_window = windows.CircularWindow(selected_radius, inner_background_radius, outer_background_radius, s.data)
for group in peak_groups:
    grouped_integrated_intensities.append(group.get_integrated_intensity(image_window) / len(group))
print("Spot intensities integrated.")

methods.plot_segmented_image(peak_groups, median_circumcentre, outer_background_radius, s)
methods.plot_group_intensities(grouped_integrated_intensities)
methods.identify_thickness(grouped_integrated_intensities, filename)
