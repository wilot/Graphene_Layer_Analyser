# Graphene Layer Analyser

## Overview

This project aims to analyse electron-microscope diffraction patterns of graphene, with the aim of being able to differentiate between monolayer and bilayer/few-layer samples. Current diffraction pattern analysis tools require significant human interaction, and are therefore time-consuming, particularly for large datasets. The necessity for human interaction also reduces reproducibility. By developing an algorithmic-based analysis tool, it is hoped that the field of graphene metrology will be improved, because layer-analysis of samples will become more approachable.

The Jupyter notebooks in the `notebooks/` directory are for testing algorithms only.

## Techniques

### Peak finding

Currently, a file is initially searched for peaks using SciPy's `peak_local_max()` method. These initial peaks are fitted to gaussians to determine a median of the full-width-half-maxima of these initial diffraction spots. With this information `peak_local_max()` is re-run using the FWHM to inform the minimum-distance filter in the peak-finder. Currently all peaks must be greater than 20% of the brightest pixel to be registered. Previously there was a gaussian filtration/dilation of the image with a sigma of one-third of the FWHM in-between the two peak-searches. This was removed because it did not appear to help the peak finding or reject noise, however if it needs to be re-added, it should probably go here...

### Peak Segmentation

Next the peaks are segmented. Initially they are segmented according to a radial distance from an estimated central position. The central position is always behind a physical beam-stop, so its location must be inferred. This has been very susceptible to a bad estimation of the central/zeroth-order spot position in the image, and getting it to work in every circumstance has been tricky. Currently an initial estimation is made by calculating the centre-of-mass of the (clipped) image. The `group_radially()` method is then run. These groups are assumed to be approaching circular due to the nature of `group_radially()`. Circumcentres of the cyclic permutations of triplets of all the spots in each radial group are computed, and the whole set of circumcentres are then median-ed to produce a new estimate of the zeroth-order spot's position. Then the radial grouping is performed again.

It is not yet known how necessary repeated re-estimation of the central-spot and repeated radial segmentation is. On a few images I have found it to be helpful, however these might be unrepresentitve. I have yet to try on the dataset I currently have with noise added in.

After grouping radially, the spots must be segmented azimuthally, to separate different twists if there is a twisted sample.

Many of the processes involved in peak segmentation and grouping are stored in the `diffraction_spots.py` module. A class for groups of spots, `SpotGroup` and methods for segmentation are stored here.

### Peak Intensity Measurement

The intensities of each peak are measured and used to differentiate between few-layer and monolayer samples. The intensity of each peak is measured seperately, however caching is used in some places and this should be made as fast as possible. Optimisation can wait for the minute though.

Intensity measurement is made using windows. For a circular window, a circular selected-area mask is formed around the targeted peak, along with an annular mask co-centred, for peak-localised background subtraction. The intensity is the summed intensity of the pixels in the window after the average background has been subtracted. This implementation is designed with extensibility in mind, should other window shapes be needed (such as a square window, for computational efficiency).

For each image a `Window` object is defined. This defines a number of things that will remain constant for all intensity calculations for spots in a particular image/diffraction-pattern. Intensity measurements are made on 'sub-windows' of the image for computational efficiency. These are just cropped areas of the original image centred on the peak of interest, generated lazily, when each peak is processed. The sub-window size is defined in the `Window` as well as the shape of each integration window around each peak.

This portion of the code works fairly well, so I don't plan on altering it or optimising it at least until the segmentation problems are ironed out.

### Error Propagation
The errors for each spot are determined by firstly estimating the pixel-error from the background pixels. The mean of the background annulus is taken, then the RMS deviation from the background-mean of all the background pixels is used. This is then squared, multiplied by the number of pixels used in the selected-area, and then square-rooted (a sum-in-quadrature). This result is taken to be the uncertainty in the intensity of that spot (see `windows.CircularWindow.get_intensity()`. The uncertainty of a SpotGroup is then taken to be the sum-in-quadrature of the uncertainties of all the spots within that SpotGroup (see `diffraction_spots.SpotGroup.calculate_integrated_intensity()`). 

Using this error-propogation approach, of the diffraction-patterns tested, the uncertainties appear to mostly be between 0.1% and 5%.

### Layer Determination

With the intensities of all the spots in the segmented groups of spots, it becomes easy to differentiate monolayer and few-layer Graphene samples. There either is or is not a factor of two difference in the intensity of the inner hexagon's spots compared to the next outer hexagon. This is currently implemented in the `methods.py` module.

## Improvements

Next, the way the results are showed needs work. Currently, debugging graphs and print statements are the only semblance of a UI. A CLI would probably be acceptable, but it really ought to be able to generate a figure for each diffraction-pattern so that the user can choose to accept or reject the thickness estimation based on the location of the identified peaks. 

Finally, the entire thing should be packaged.