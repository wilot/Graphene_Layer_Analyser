"""Contains configuration variables and functions to process a diffraction pattern

"""

import numpy as np
import skimage.feature
import skimage.filters
import skimage.morphology
from hyperspy._signals.signal2d import Signal2D

import diffraction_spots
import windows
import methods


class ConfigurableParameters:
    """Contains an initialiser for the basic, tuneable parameters required for the processing of diffraction-patterns.

    Intended to make parameter tuning more convenient, all important tunable parameters for the processing of
    diffraction-patterns should be stored here. """

    def __init__(self, peak_fwhm, image_axis_scale):
        # self.selected_radius = max(peak_fwhm, 8.)
        self.selected_radius = peak_fwhm
        self.inner_background_radius = self.selected_radius * 2
        self.outer_background_radius = self.selected_radius * 2.5

        # Image dilation width
        self.dilation_sigma = self.selected_radius
        # Spot minimum separation
        self.spot_min_separation = int(self.outer_background_radius * 1.5) * 2
        # A threshold below which data is ignored in various methods
        self.image_low_threshold = 100
        self.image_upper_threshold = 200
        # A radius from the centre of the image that the central spot can be assumed to be within
        self.centre_within_radius = 200
        # Radial segmentation threshold of 0.2/nm
        self.radial_group_threshold = 2 / image_axis_scale
        if not (5 < self.radial_group_threshold < 50):  # Usually due to bad metadata
            self.radial_group_threshold = self.selected_radius * 2.5  # Bit buggy, really ought to have the metadata!
        # Angular segmentation threshold (in degrees)
        self.angular_group_threshold = 7.5


class ProcessedResult:
    """Packages the results of the processing of a diffraction-pattern, including group intensities and result flags.

    Contains the results of the processing of the diffraction-pattern, along with whether the processing was
    deemed a success and the determined thickness of the Graphene sample. Note, a figure should be returned even if
    the analysis is deemed a failure. If a figure cannot be returned, an exception should be raised instead.

    Also contains useful debugging information.
    """

    def __init__(self):
        self.monolayer = None  # To contain the result
        self._initially_identified_spots = None
        self.insufficient_spots = False
        self.central_spot_failure = False
        self.insufficient_spot_groups = False
        self.spot_groups = None
        self.central_spot_coordinate = None
        self.outer_background_radius = None
        self.diffraction_pattern = None
        self.peak_groups_numbers = {"centrally unrefined": -1,
                                    "centrally refined": -1,
                                    "azimuthally grouped": -1}

    def is_success(self):
        return not (self.central_spot_failure or self.insufficient_spots or self.insufficient_spot_groups)

    @property
    def initially_identified_spots(self):
        return self._initially_identified_spots

    @initially_identified_spots.setter
    def initially_identified_spots(self, num_spots: int):
        self.insufficient_spots = num_spots < 10
        self._initially_identified_spots = num_spots

    def __repr__(self):  # In case you need to print this, for debugging...
        this_object = "Result printout:\n"
        this_object += f"{self.monolayer=}\n"
        this_object += f"{self._initially_identified_spots=}\n"
        this_object += f"{self.insufficient_spots=}\n"
        this_object += f"{self.central_spot_failure=}\n"
        this_object += f"len(spot_groups)={len(self.spot_groups)}\n"
        for k, v in self.peak_groups_numbers.items():
            this_object += f"Peak group numbers ({k}): {v}\n"
        this_object += f"{self.central_spot_coordinate=}\n"
        this_object += f"{self.outer_background_radius=}\n"
        this_object += f"success: {self.is_success()}\n"
        return this_object


def primary_peak_search(pattern: Signal2D, dilation_sigma, min_distance, rel_min=0.05):
    """Performs the primary peak-finding, before peaks are filtered by radial and polar position."""

    # TODO: Find a more computationally efficient filter
    dilated_image = skimage.filters.gaussian(pattern.data, sigma=dilation_sigma)
    # mean_mask = skimage.morphology.disk(dilation_sigma)
    # dilated_image = skimage.filters.rank.mean(pattern.data, mean_mask)
    peaks_list = skimage.feature.peak_local_max(dilated_image, min_distance=min_distance,
                                                threshold_rel=rel_min, exclude_border=True)
    del dilated_image  # Be extra sure that this is being garbage collected asap!
    return peaks_list


def refined_peak_radial_segmentation(peaks_list, peak_groups, params, central_peak_estimate, result):
    centrally_unrefined_peak_groups = peak_groups
    circumcentres = []

    for peak_group in peak_groups:
        if len(peak_group) > 2:
            try:
                group_circumcentres = peak_group.get_circumcentres(params.centre_within_radius, central_peak_estimate)
            except RuntimeError:
                continue  # Failed to find any circumcentres for this group
            circumcentres.extend(group_circumcentres)
    if len(circumcentres) > 0:
        central_spot = np.median(np.array(circumcentres), axis=0)
        peak_groups = diffraction_spots.group_radially(central_spot, peaks_list,
                                                       threshold=params.radial_group_threshold)
        peak_groups = diffraction_spots.prune_spot_groups(peak_groups, min_spots=3)
        centrally_refined_peak_groups = peak_groups
        if len(centrally_refined_peak_groups) < 2:  # Central refinement might have actually made things worse...
            peak_groups = centrally_unrefined_peak_groups
        result.peak_groups_numbers["centrally refined"] = len(centrally_refined_peak_groups)
    else:
        central_spot = central_peak_estimate
        result.central_spot_failure = True
        print(f"\tCentral spot location estimation unreliable")

    if len(peak_groups) < 2:  # Not much point is there :(
        result.insufficient_spot_groups = True
        result.spot_groups = []
        result.central_spot_coordinate = central_spot

    return central_spot, peak_groups


def process(pattern: Signal2D) -> ProcessedResult:
    """Processes a diffraction pattern, analysing the spots and returning the results."""

    result = ProcessedResult()
    if not pattern.axes_manager.navigation_axes == 0:  # The image is sneakily an image stack!
        pattern = pattern.mean()
    pattern.data = np.nan_to_num(pattern.data)
    result.diffraction_pattern = pattern
    pattern_size = pattern.data.shape[0], pattern.data.shape[1]

    # Here I add a beam-stop because that makes it more resilient.
    mask_grid_x, mask_grid_y = np.mgrid[0: pattern_size[0], 0: pattern_size[1]]
    centre = pattern_size[0] // 2, pattern_size[1] // 2
    mask_grid_x -= centre[0]
    mask_grid_y -= centre[1]
    radial_distances = np.sqrt(mask_grid_x ** 2 + mask_grid_y ** 2)
    mask_radius = int(0.15 * pattern_size[0])
    mask = radial_distances < mask_radius
    pattern.data[mask] = np.median(pattern.data.flatten())
    peaks_list = skimage.feature.peak_local_max(pattern.data, min_distance=int(pattern_size[0] / 50),
                                                threshold_rel=0.1, exclude_border=True, num_peaks=20)

    peak_fwhm = np.median([methods.get_gaussian_fwhm(pattern.data, peak) for peak in peaks_list])
    # print(f"Detected peak fwhm {peak_fwhm}")

    # Configure tunable parameters
    params = ConfigurableParameters(peak_fwhm, pattern.axes_manager[0].scale)

    result.outer_background_radius = params.outer_background_radius

    # Primary peak-finding
    peaks_list = primary_peak_search(pattern, params.dilation_sigma, params.spot_min_separation//3)
    # print(f"\t{len(peaks_list)} initially identified spots.")
    if len(peaks_list) > 40:  # Usually indicates insufficient image dilation, give it one last go
        print(f"\tToo many spots detected with initial image-dilation setting, assuming this is due to insufficient "
              f"noise and re-attempting (▼ file) with dilation {params.dilation_sigma * 0.25}px and minimum separation "
              f"{int(pattern_size[0] * 2.5E-2)}px.")
        peaks_list = primary_peak_search(pattern, params.dilation_sigma * 0.25, int(pattern_size[0] * 2.5E-2), 0.15)
        params.radial_group_threshold *= 3  # Relax segmentation
        params.angular_group_threshold *= 1
        print(f"{params.radial_group_threshold=}")
    elif len(peaks_list) < 10:
        print(f"\tToo few spots detected with initial image-dilation setting, assuming this is due to noise and "
              f"re-attempting (▼ file) with dilation {params.dilation_sigma * 3}px and minimum separation "
              f"{int(pattern_size[0] * 2.5E-2)}px.")
        peaks_list = primary_peak_search(pattern, params.dilation_sigma * 3, int(pattern_size[0] * 2.5E-2))
        params.radial_group_threshold *= 2  # Relax segmentation
        params.angular_group_threshold *= 1
    result.initially_identified_spots = len(peaks_list)

    # Segment spots radially initially just to help determine central spot's position, then once more to classify the
    # spots into the correct groups.
    # print(f"Initial radial grouping, with radial group threshold {params.radial_group_threshold*5}")
    central_spot = methods.centre_of_mass(pattern.data,
                                          lower_threshold_abs=params.image_low_threshold,
                                          upper_threshold_abs=params.image_upper_threshold)
    peak_groups = diffraction_spots.group_radially(central_spot, peaks_list,
                                                   threshold=params.radial_group_threshold * 5)
    centrally_unrefined_peak_groups = peak_groups
    result.peak_groups_numbers["centrally unrefined"] = len(centrally_unrefined_peak_groups)

    central_spot, peak_groups = refined_peak_radial_segmentation(peaks_list, peak_groups, params, central_spot, result)

    # Angle subtended by a spot in the innermost ring
    subtended_angle = np.arcsin((params.outer_background_radius * 2) /
                                np.linalg.norm(peak_groups[0].spots[0] - central_spot))
    subtended_angle = np.rad2deg(subtended_angle)
    if params.angular_group_threshold * 0.5 > subtended_angle:
        params.angular_group_threshold = params.angular_group_threshold * 0.5
    elif subtended_angle > params.angular_group_threshold * 1.5:
        params.angular_group_threshold = params.angular_group_threshold * 1.5
    elif np.isnan(subtended_angle):
        params.angular_group_threshold = params.angular_group_threshold * 1.5
    else:
        params.angular_group_threshold = subtended_angle

    # Now segment according to polar-angle too
    peak_groups = diffraction_spots.group_azimuthally(central_spot, peak_groups,
                                                      threshold=params.angular_group_threshold)
    print(f"Azimuthal segmentation performed with {params.angular_group_threshold=}°, while subtending angle "
          f"{subtended_angle}°")

    peak_groups_pruned = diffraction_spots.prune_spot_groups(peak_groups, min_spots=4)
    if len(peak_groups_pruned) > 1:
        peak_groups = peak_groups_pruned
    result.peak_groups_numbers["azimuthally grouped"] = len(peak_groups)

    image_window = windows.CircularWindow(params.selected_radius, params.inner_background_radius,
                                          params.outer_background_radius, pattern.data)
    for group in peak_groups:
        group.calculate_integrated_intensity(image_window)

    result.spot_groups = peak_groups
    result.central_spot_coordinate = central_spot
    result.insufficient_spot_groups = len(peak_groups) < 2

    # DEBUGGING: for displaying the image window's masks on some random spot.
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # image_window.show_masks(*peak_groups[2].spots[0], ax)
    # plt.show()

    # print(result)  # For debugging

    return result
