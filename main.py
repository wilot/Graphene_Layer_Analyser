"""Controller module

This module links the user interface and processing parts of the program. Parallelized processing and such should be
dealt with here. """

import user_interface as ui
import pattern_processor


filenames, output_dir, show, parallelize = ui.get_cli_arguments()

if parallelize:
    print("Parallelization not implemented yet, continuing serially.")

for filename, signal in zip(filenames, ui.load_signals(filenames)):
    process_result = pattern_processor.process(signal)
    figure = ui.plot(filename, process_result, show)
    ui.save_figure(figure, output_dir, filename)

print("Success!")
