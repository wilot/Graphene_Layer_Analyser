"""Utility methods."""

import numpy as np
import scipy.optimize
import scipy.ndimage
import matplotlib.pyplot as plt


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


def _gaussian(pos, xo, yo, sigma, amplitude, offset):
    """Returns a 2D gaussian function flattened to a 1D array."""

    g = offset + amplitude * np.exp(
        - (((pos[0] - xo) ** 2) / (2 * sigma ** 2) + ((pos[1] - yo) ** 2) / (2 * sigma ** 2))
    )

    return g.ravel()
