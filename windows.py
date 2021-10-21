"""Contains windows for images.

Contains classes and functions defining and processing windows within images. Windows are selected areas over which
analysis or manipulation might occur. An example might be a circular window around a point, to integrate the
intensity of that point. Windows of background-subtraction would also need to be defined, in this example an annulus
around the circular window.

Defines "Masks" and "Windows". A Mask is a simple class that defines an area, which can be applied to an image or set
of images. A window uses masks to calculate intensities using background subtraction. For example; a CircularWindow
uses a Circular *and* an annular mask (for summation and background subtraction respectively). For the sake of
computational efficiency, mask computations are performed on a sub-window of the original image.

Note: Currently, pixel smoothing of mask boundaries is ignored, so mask edges will be spiky!"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Tuple, Any, Union, List

import matplotlib.axes
import numpy as np


class Window(ABC):
    """An interface defining all Windows. Window define a photometry window within a parent image. This window is
    then applied at a set of coordinates as specified upon calls to Window.get_intensity(). """

    def __init__(self, sub_window_radial_extent_x: int, sub_window_radial_extent_y: int, parent_image: np.ndarray):
        self.sub_window_radial_extents = (sub_window_radial_extent_x, sub_window_radial_extent_y)
        self.parent_image = parent_image

    def get_sub_window(self, target: Tuple[float, float]) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Returns a sub-window of the Window's parent image, along with the new target coordinates x, y."""
        sub_window_slices = self._generate_sub_window_extent(target)
        sub_window_centre_px = (target[0] - sub_window_slices[0].start,
                                target[1] - sub_window_slices[1].start)
        sub_window = self.parent_image[sub_window_slices]
        return sub_window, sub_window_centre_px

    def _generate_sub_window_extent(self, centre: tuple) -> Tuple[slice]:
        """Determines the start & end indices (in x & y) of a rectangular sub-window in the parent image, centred on the
        window's target coordinates. """

        start_x = centre[0] - self.sub_window_radial_extents[0]
        start_y = centre[1] - self.sub_window_radial_extents[1]
        end_x = centre[0] + self.sub_window_radial_extents[0]
        end_y = centre[1] + self.sub_window_radial_extents[1]
        start_x, start_y, end_x, end_y = map(int, (start_x, start_y, end_x, end_y))
        start_x, start_y = max(start_x, 0), max(start_y, 0)
        end_x, end_y = min(end_x, self.parent_image.shape[0]), min(end_y, self.parent_image.shape[1])
        return np.s_[start_x:end_x, start_y:end_y]

    @abstractmethod
    def get_integrated_intensity(self, target_x: float, target_y: float) -> float: pass

    @abstractmethod
    def get_average_intensity(self, target_x: float, target_y: float) -> float: pass

    @abstractmethod
    def get_maximum_intensity(self, target_x: float, target_y: float) -> float: pass

    @abstractmethod
    def get_intensity(self, target_x: float, target_y: float, method: str = "None") -> float: pass


class Mask(ABC):
    """An interface defining all masks"""

    def __init__(self, sub_window_shape: Tuple, centre: Tuple[float, float]):
        self.sub_window_shape = sub_window_shape
        self.centre = centre
        self.mask, self.num_px = self.generate_mask()

    @abstractmethod
    def generate_mask(self) -> Tuple[np.ndarray, np.int64]: raise NotImplementedError

    @staticmethod
    @lru_cache(maxsize=1)
    def cached_mesh_grid(sub_window_shape: tuple):
        return np.mgrid[0: sub_window_shape[0], 0: sub_window_shape[1]].astype("float64")

    def get_sum(self, sub_window: np.ndarray, background_value: float = 0):
        background_subtracted_window = sub_window - background_value
        masked_sum = np.sum(background_subtracted_window[self.mask])
        return float(masked_sum)

    def get_mean(self, sub_window: np.ndarray, background_value: float = 0):
        background_subtracted_window = sub_window - background_value
        masked_mean = np.mean(background_subtracted_window[self.mask])
        return float(masked_mean)

    def get_max(self, sub_window: np.ndarray, background_value: float = 0):
        background_subtracted_window = sub_window - background_value
        masked_max = np.max(background_subtracted_window[self.mask])
        return float(masked_max)

    def get_rms_error(self, sub_window, target_value):
        """Computes the the RMS deviation of pixels in the sub-window away from the target value over the
        mask. """
        deviations = sub_window[self.mask] - target_value
        rms = np.sqrt(np.mean(deviations ** 2))  # The RMS deviation
        return rms

    def __del__(self):
        del self.mask


class CircularMask(Mask):
    """Defines a circular mask."""

    def __init__(self, radius: float, sub_window_shape: tuple, centre: Tuple[float, float]):
        """Generates a circular mask for an image with dimensions defined by sub_window_shape. """

        self.radius = radius
        super(CircularMask, self).__init__(sub_window_shape, centre)

    def generate_mask(self) -> Tuple[np.ndarray, np.int64]:
        # Caching saves a bit of time, I hope...
        xx, yy = super(CircularMask, self).cached_mesh_grid(self.sub_window_shape).copy()
        xx -= self.centre[0]
        yy -= self.centre[1]
        xx_squared = xx ** 2
        yy_squared = yy ** 2
        mask = (xx_squared + yy_squared) < self.radius ** 2
        return mask, mask.sum()


class AnnularMask(Mask):
    """Defines an annular mask"""

    def __init__(self, inner_radius: float, outer_radius: float, sub_window_shape: Tuple,
                 centre: Tuple[float, float]):
        """Generates an annular mask for an image or set of images, of dimensions defined by masked_image_shape."""

        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.centre = centre
        super(AnnularMask, self).__init__(sub_window_shape, centre)

    def generate_mask(self) -> Tuple[np.ndarray, np.int64]:
        inner_mask = CircularMask(self.inner_radius, self.sub_window_shape, self.centre)
        outer_mask = CircularMask(self.outer_radius, self.sub_window_shape, self.centre)
        mask = np.logical_xor(outer_mask.mask, inner_mask.mask)
        return mask, mask.sum()


class CircularWindow(Window):
    """Defines a circular window template for a specific image. Template has a circular selected area with annular
    background subtraction. The coordinates of the targets are then passed to the get-intensity methods, which then
    apply the window template to the targeted coordinates. """

    def __init__(self, selected_radius: float, background_inner_radius: float, background_outer_radius: float,
                 parent_image: np.ndarray):
        """Generates a circular window with a selected circular area (without boundary smoothing/interpolation
        applied) defined by selected_radius, and uses an annulus for background subtraction as defined by
        background_inner_radius and background_outer_radius. """

        self.selected_radius = selected_radius
        self.background_inner_radius = background_inner_radius
        self.background_outer_radius = background_outer_radius
        # A 10% boundary between the background annulus and the sub-window boundary seems sensible...
        sub_window_radial_extent_x = sub_window_radial_extent_y = int(background_outer_radius * 1.1)
        super(CircularWindow, self).__init__(sub_window_radial_extent_x, sub_window_radial_extent_y, parent_image)

    def get_intensity(self, target_x: float, target_y: float, method: str = "None") -> Tuple[float, Union[float, None]]:
        """Determines the intensity (and its uncertainty) from the pre-configured CircularWindow,
        at the provided target coordinates. Note, standard deviation is only implemented for the sum method so far...
        """

        sub_image, selected_area, background = self._generate_masks((target_x, target_y))
        background_average = background.get_mean(sub_image)
        background_error = background.get_rms_error(sub_image, background_average)  # Background RMS deviation

        if method == "integrated" or "None":
            sa_sum = selected_area.get_sum(sub_image, background_value=background_average)
            sa_error = background_error * selected_area.num_px
            uncertainty = sa_error
            return sa_sum, uncertainty
        elif method == "average":
            return selected_area.get_mean(sub_image, background_value=background_average), None
        elif method == "maximum":
            return selected_area.get_max(sub_image, background_value=background_average), None
        else:
            raise ValueError("The 'method' passed to CircularWindow.get_intensity() was not recognised.")

    def get_integrated_intensity(self, target_x: float, target_y: float) -> Tuple[float, Union[float, None]]:
        return self.get_intensity(target_x, target_y, "integrated")

    def get_average_intensity(self, target_x: float, target_y: float) -> Tuple[float, Union[float, None]]:
        return self.get_intensity(target_x, target_y, "average")

    def get_maximum_intensity(self, target_x: float, target_y: float) -> Tuple[float, Union[float, None]]:
        return self.get_intensity(target_x, target_y, "maximum")

    def _generate_masks(self, target: Tuple[float, float]) -> Tuple[np.ndarray, CircularMask, AnnularMask]:
        sub_window, sub_window_target = self.get_sub_window(target)
        selected_area = CircularMask(self.selected_radius, sub_window.shape, sub_window_target)
        background = AnnularMask(self.background_inner_radius, self.background_outer_radius, sub_window.shape,
                                 sub_window_target)
        return sub_window, selected_area, background

    def show_masks(self, target_x: float, target_y: float, ax: matplotlib.axes.Axes, print_debug=False) \
            -> None:
        """For debugging, return an array indicating the masks within the sub-window"""
        sub_window, sub_window_target = self.get_sub_window((target_x, target_y))
        selected_area = CircularMask(self.selected_radius, sub_window.shape, sub_window_target)
        background = AnnularMask(self.background_inner_radius, self.background_outer_radius, sub_window.shape,
                                 sub_window_target)
        mask_image = np.zeros(sub_window.shape)
        mask_image[selected_area.mask] = 2
        mask_image[background.mask] = 1
        sub_image = self.parent_image[self._generate_sub_window_extent((target_x, target_y))]
        if print_debug:
            print("selected_area mask shape:", selected_area.mask.shape)
            print("background mask shape:", background.mask.shape)
            print("sub-image shape:", sub_image.shape)

        from matplotlib import colors
        color_map = colors.ListedColormap(["white", "green", "red"])
        norm = colors.BoundaryNorm([0, 1, 2, 3], color_map.N)
        ax.imshow(sub_image, cmap="Greys")
        ax.imshow(mask_image, alpha=0.3, cmap=color_map, norm=norm)
        ax.set_title("Image window spot masks")

        # return mask_image, sub_image
