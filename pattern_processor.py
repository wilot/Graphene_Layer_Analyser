"""Contains configuration variables and functions to process a diffraction pattern

"""

import numpy as np
import skimage.feature
import skimage.filters
from hyperspy._signals.signal2d import Signal2D

import diffraction_spots
import windows
import methods


class WindowParameters:
    """Contains an initialiser for the basic, tuneable parameters required for the processing of diffraction-patterns.

    Intended to make parameter tuning more convenient, all important tunable parameters for the processing of
    diffraction-patterns should be stored here. """

    def __init__(self, peak_fwhm, image_axis_scale):
        self.selected_radius = peak_fwhm * 2
        self.inner_background_radius = peak_fwhm * 4
        self.outer_background_radius = peak_fwhm * 5

        # Image dilation width
        self.dilation_sigma = self.selected_radius
        # Spot minimum separation
        self.spot_min_separation = int(self.outer_background_radius * 1.5)
        # A threshold below which data is ignored in various methods
        self.image_low_threshold = 100
        # A radius from the centre of the image that the central spot can be assumed to be within
        self.centre_within_radius = 200
        # Radial segmentation threshold of 0.2/nm
        self.radial_group_threshold = max(0.2 / image_axis_scale, 100)


class ProcessedResult:
    """Packages the results of the processing of a diffraction-pattern, including group intensities and result flags.

    Contains the results of the processing of the diffraction-pattern, along with whether or not the processing was
    deemed a success and the determined thickness of the Graphene sample. Note, a figure should be returned even if
    the analysis is deemed a failure. If a figure cannot be returned, an exception should be raised instead."""

    def __init__(self):
        self.monolayer = None  # To contain the result
        self._initially_identified_spots = None
        self.insufficient_spots = False
        self.central_spot_failure = False
        self.spot_groups = None
        self.central_spot_coordinate = None
        self.outer_background_radius = None
        self.diffraction_pattern = None

    def is_success(self):
        return not (self.central_spot_failure or self.insufficient_spots)

    @property
    def initially_identified_spots(self):
        return self.initially_identified_spots

    @initially_identified_spots.setter
    def initially_identified_spots(self, num_spots: int):
        self.insufficient_spots = num_spots < 10
        self._initially_identified_spots = num_spots


def primary_peak_search(pattern: Signal2D, dilation_sigma, min_distance):
    """Performs the primary peak-finding, before peaks are filtered by radial and polar position."""

    # TODO: Find a more computationally efficient filter
    dilated_image = skimage.filters.gaussian(pattern.data, sigma=dilation_sigma)
    peaks_list = skimage.feature.peak_local_max(dilated_image, min_distance=min_distance,
                                                threshold_rel=0.05, exclude_border=True)
    return peaks_list


def process(pattern: Signal2D) -> ProcessedResult:
    """Processes a diffraction pattern, analysing the spots and returning the results."""

    result = ProcessedResult()
    result.diffraction_pattern = pattern

    # First identify any peak, measure its Gaussian FWHM, filter the image according to this and re-detect peaks.
    peaks_list = skimage.feature.peak_local_max(pattern.data, min_distance=100, threshold_rel=0.2, exclude_border=True,
                                                num_peaks=5)
    peak_fwhm = np.median([methods.get_gaussian_fwhm(pattern.data, peak) for peak in peaks_list])

    # Configure tunable parameters
    params = WindowParameters(peak_fwhm, pattern.axes_manager[0].scale)

    result.outer_background_radius = params.outer_background_radius

    # Primary peak-finding
    peaks_list = primary_peak_search(pattern, params.dilation_sigma, params.spot_min_separation)
    # TODO: Decide whether this is needed; seems a bit clunky, probably a bad idea...
    # if len(peaks_list) > 30:  # Usually indicates insufficient image dilation, give it one last go
    #     print("Too many spots, re-dilating!")
    #     peaks_list = primary_peak_search(pattern, params.dilation_sigma * 2, params.spot_min_separation)
    # elif len(peaks_list) < 10:
    #     print("Too few spots, re-dilating!")
    #     peaks_list = primary_peak_search(pattern, params.dilation_sigma * 0.6, params.spot_min_separation)
    result.initially_identified_spots = len(peaks_list)

    # Estimate the position of the zeroth-order diffraction spot by segmenting spots by radial distance from the image's
    # centre-of-mass and then taking the median circumcentre of all permutations of spots within each group. Use this
    # refined circumcentral zeroth-order spot position to refine radial segmentation.
    central_peak_estimate = methods.centre_of_mass(pattern.data,
                                                   lower_threshold_abs=params.image_low_threshold)
    peak_groups = diffraction_spots.group_radially(central_peak_estimate, peaks_list,
                                                   threshold=params.radial_group_threshold)
    peak_groups = diffraction_spots.prune_spot_groups(peak_groups, min_spots=3)
    circumcentres = []
    for peak_group in peak_groups:
        try:
            group_circumcentres = peak_group.get_circumcentres(permitted_radius=params.centre_within_radius)
        except RuntimeError: continue  # Failed to find any circumcentres :(
        else: circumcentres.extend(group_circumcentres)
    if len(circumcentres) > 0:
        central_spot = np.median(np.array(circumcentres), axis=0)
        peak_groups = diffraction_spots.group_radially(central_spot, peaks_list,
                                                       threshold=params.radial_group_threshold)
    else:
        central_spot = central_peak_estimate
        result.central_spot_failure = True

    # Now segment according to polar-angle too
    peak_groups = diffraction_spots.group_azimuthally(central_spot, peak_groups, threshold=2)

    peak_groups = diffraction_spots.prune_spot_groups(peak_groups)

    image_window = windows.CircularWindow(params.selected_radius, params.inner_background_radius,
                                          params.outer_background_radius, pattern.data)
    for group in peak_groups:
        group.calculate_integrated_intensity(image_window)

    result.spot_groups = peak_groups
    result.central_spot_coordinate = central_spot

    # DEBUGGING: for displaying the image window's masks on some random spot.
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # image_window.show_masks(*peak_groups[0].spots[1], ax)
    # plt.show()

    return result
