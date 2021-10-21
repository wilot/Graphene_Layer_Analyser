"""Test module

This module will primarily be used for ironing out the analytical techniques. Not to be used by users.
"""

import hyperspy.api as hs

import user_interface as ui
import pattern_processor


# ui.get_cli_arguments()

# For debugging only, to be superseded by get_cli_arguments(). Modify with your own path to data while developing.
image_filenames = ui.load_filenames()
filename = image_filenames[0]["dataset_file_list"][17]
s = hs.load(filename)
print("Image loaded")

process_result = pattern_processor.process(s)
ui.assign_spot_group_colours(process_result.spot_groups)
ui.plot_segmented_image(process_result.spot_groups,
                        process_result.central_spot_coordinate,
                        process_result.outer_background_radius,
                        process_result.diffraction_pattern)
ui.plot_group_intensities(process_result.spot_groups)
print("Success!")