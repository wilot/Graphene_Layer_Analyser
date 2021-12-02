"""A class and methods for the segmentation of identified spots into groups of spots."""

from typing import Union, List

import numpy as np

import windows


class SpotGroup:
    """A class for containing a group of diffraction spot coordinates and performing operations on them."""

    def __init__(self, spots: np.ndarray):
        """Takes a list of spot coordinates in [[x, y], ...]"""

        self.spots = spots
        self.colour = None
        self.group_intensity = 0
        self.group_uncertainty = 0

    def append(self, spot: np.ndarray):
        """Appends a spot or array of spots."""

        # If spot is singular, add a dimension to append
        if spot.shape == (2,):
            self.spots = np.append(self.spots, (spot,))
            return
        # Otherwise, can append the list as is
        self.spots = np.append(self.spots, spot)

    def get_circumcentres(self, permitted_radius) -> Union[np.ndarray, None]:
        """Returns a list of all the circumcentres (before a median is taken). For debugging."""

        if len(self.spots) < 3:
            raise RuntimeError("Not enough spots to calculate circumcentres.")

        triplets = self._generate_triplet_combinations()
        circumcentres = np.array([self._get_circumcentre(triplet) for triplet in triplets])
        central_filter = lambda pos: not any(np.isnan(pos)) and np.linalg.norm(pos - (2048, 2048)) < permitted_radius
        circumcentres = [circ for circ in circumcentres if central_filter(circ)]

        if len(circumcentres) < 1:
            raise RuntimeError("No valid circumcentres in SpotGroup.")

        return np.array(circumcentres)

    def calculate_integrated_intensity(self, image_window: windows.Window):
        """Returns the summed, background corrected intensities of all spots in the group."""

        intensities = []
        uncertanties = []
        for spot in self.spots:
            intensity, uncertainty = image_window.get_integrated_intensity(*spot)
            intensities.append(intensity)
            if uncertainty is not None:
                uncertanties.append(uncertainty)

        self.group_intensity = sum(intensities) / len(self)
        if len(uncertanties) == 0:
            return
        uncertanties = np.array(uncertanties)
        uncertainty = np.sqrt(np.sum(uncertanties**2))
        uncertainty /= len(self)
        self.group_uncertainty = uncertainty

    def __len__(self):
        return len(self.spots)

    def _generate_triplet_combinations(self):
        """Generates a list of different triplet permutations of this group's spots."""

        triplets = []
        indices = np.arange(len(self.spots))
        for _ in range(len(self.spots)):
            triplets.append([self.spots[i] for i in indices[:3]])
            indices = np.roll(indices, shift=1)

        return triplets

    # noinspection DuplicatedCode,PyPep8Naming
    @staticmethod
    def _get_circumcentre(triplet):
        """Calculates the circumcentre of a triplet of coordinates using linear algebra."""

        A, B, C = triplet
        Ax, Ay = A
        Bx, By = B
        Cx, Cy = C

        D = 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))

        Ux = ((Ax ** 2 + Ay ** 2) * (By - Cy) + (Bx ** 2 + By ** 2) * (Cy - Ay) + (Cx ** 2 + Cy ** 2) * (Ay - By)) / D
        Uy = ((Ax ** 2 + Ay ** 2) * (Cx - Bx) + (Bx ** 2 + By ** 2) * (Ax - Cx) + (Cx ** 2 + Cy ** 2) * (Bx - Ax)) / D

        return np.array((Ux, Uy))


def group_radially(centre_position, positions: np.ndarray, threshold, verbose=False) -> List[SpotGroup]:
    """Groups spot positions by radial distance and returns a list of spot groups. Parameter types deliberately force
    you to group radially *before* grouping azimuthally, by design.

    This function needs work and is frequently updated... Too susceptible to a bad estimate of the centre position."""

    if centre_position is None:
        raise ValueError("Centre position must not be None")

    # First compute the distances to this central spot
    distances_to_central_spot = np.linalg.norm([position - centre_position for position in positions], axis=1)
    distances_to_central_spot = np.expand_dims(distances_to_central_spot, axis=1)

    # Create [[x, y, radial_distance], ...] array and sort by distance
    position_distance = np.concatenate((positions, distances_to_central_spot), axis=1)
    position_distance = position_distance[position_distance[:, 2].argsort()]

    # Compute the step-length to each next radial distance
    radial_distance_steps = np.diff(position_distance[:, 2])

    if verbose:  # For debugging
        print("Threshold:", threshold)
        print(position_distance[:, 2])
        print(radial_distance_steps)

    # Find the position_distance indices *after* a big increase in radial distance
    big_step_indices = [0]
    for index, radial_step in enumerate(radial_distance_steps, start=1):
        if radial_step > threshold:  # 10% the distance from the centre to the first spot
            big_step_indices.append(index)

    # Create position groups
    groups = []

    if len(big_step_indices) == 1:  # This means radial segmentation failed :(
        this_group = SpotGroup(position_distance[:, :2])
        groups.append(this_group)
        return groups

    for prev_step_index, step_index in zip(big_step_indices, big_step_indices[1:]):
        this_group_positions = position_distance[prev_step_index:step_index, :2]
        this_group = SpotGroup(this_group_positions)
        groups.append(this_group)
    groups.append(
        SpotGroup(position_distance[big_step_indices[-1]:, :2])
    )

    return groups


def group_azimuthally(centre_position: List[float],
                      spot_groups: Union[SpotGroup, List[SpotGroup]],
                      threshold: float) -> List[SpotGroup]:
    """Groups spots azimuthally, once they have already been grouped radially. Where the spots are greater than
    "threshold" degrees away from 60° periodic the spot group will be further segmented.

    Works by using the centre position to determine the polar/azimuthal coordinate of each spot in each group. This
    is then wrapped around 60°. The threshold is used to divide the 60° into segments identified by an index, and the
    spots are grouped by which segment they fall into. Their wrapped angular position relative to a reference wrapped
    spot (the median wrapped angle) is used to make it unlikely that the boundary of a segment falls within a valid
    cluster.

    Parameters
    ----------
    centre_position : The (x, y) coordinate of the centre about which polar coordinates are generated.
    spot_groups : A list of spot groups, each of which will be further segmented.
    threshold : The angular segmentation threshold (in degrees).
    """

    output_spot_groups = []
    for spot_group in spot_groups:

        if len(spot_group) < 2:
            output_spot_groups.append(spot_group)
            continue

        y_coordinates = spot_group.spots[:, 1] - centre_position[1]
        x_coordinates = spot_group.spots[:, 0] - centre_position[0]
        polar_angles = np.rad2deg(np.arctan(y_coordinates / x_coordinates))  # arctan(Y / X)

        # Graphene's diffraction pattern shows 6-fold radial symmetry (periodic every 60°)
        wrapped_polar_angles = polar_angles % 60
        # Compute the deviations from perfect 6-fold symmetry, wrapping around 60°
        # wrapped_angle_distance = lambda a, b: min(abs(b - a), abs(abs(b - a) - 60.))
        angle_distance = lambda a, b: b - a
        reference_angle = np.mean(wrapped_polar_angles) - threshold/4.
        angle_differences = [angle_distance(reference_angle, b) for b in wrapped_polar_angles]

        # Segment
        segmented_group = {}
        for index, diff in enumerate(angle_differences):
            segmented_group_key = int(diff // threshold)
            if segmented_group_key not in segmented_group.keys():
                segmented_group[segmented_group_key] = [spot_group.spots[index], ]
            else:
                segmented_group[segmented_group_key].append(spot_group.spots[index])
        for key, spot_positions in segmented_group.items():
            spot_positions = np.array(spot_positions)
            new_spot_group = SpotGroup(spot_positions)
            output_spot_groups.append(new_spot_group)

    return output_spot_groups


def prune_spot_groups(spot_groups: List[SpotGroup], min_spots=4):
    """Removes spot groups containing fewer than min_spots."""

    keep_group = lambda spot_group: len(spot_group) >= min_spots
    pruned_spot_groups = list(filter(keep_group, spot_groups))
    return pruned_spot_groups
