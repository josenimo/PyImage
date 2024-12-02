#system
from loguru import logger
import argparse
import sys
import os
import time
#imports
import skimage.segmentation as segmentation
import skimage.io as io
import numpy as np


def get_args():
    """ Get arguments from command line """
    description = """Expand labeled masks by a certain number of pixels."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-r", "--input",    dest="input", action="store", required=True, help="File path to input mask or folders with many masks")
    inputs.add_argument("-o", "--output",   dest="output", action="store", required=True, help="Path to output mask, or folder where to save the output masks")
    inputs.add_argument("-p", "--pixels",   dest="pixels", action="store", type=int, required=False, help="Image pixel size")
    inputs.add_argument("-l", "--log-level",dest="loglevel", default='INFO', choices=["DEBUG", "INFO"], help='Set the log level (default: INFO)')
    arg = parser.parse_args()
    arg.input = os.path.abspath(arg.input)
    arg.output = os.path.abspath(arg.output)
    arg.pixels = int(arg.pixels)
    return arg

def check_input_outputs(args):
    """ check if input is a file or a folder """
    #input
    assert os.path.exists(args.input), "Input must exist"
    assert os.path.isfile(args.input), "Input must be a file or a folder"
    assert args.input.endswith(".tif"), "Input file must be a .tif file"
    #output
    assert args.output.endswith(".tif"), "Output file must be a .tif file"
    #pixels
    assert args.pixels > 0, "Pixels must be a positive integer"
    
def expand_mask(input_path:str, output_path:str, how_many_pixels:int, type_of_input:str):
    """ Expand all masks in a folder by a certain number of pixels """
    logger.info(f"Processing {input_path}")
    label = io.imread(os.path.join(input_path))
    logger.info(f"Label shape: {label.shape}, Label data type: {label.dtype}")
    expanded_labels = segmentation.expand_labels(label, how_many_pixels)
    max_value = expanded_labels.max()

    # set data type dynamically
    if max_value <= np.iinfo(np.uint8).max:
        expanded_labels = expanded_labels.astype('uint8')
    elif max_value <= np.iinfo(np.uint16).max:
        expanded_labels = expanded_labels.astype('uint16')
    elif max_value <= np.iinfo(np.uint32).max:
        expanded_labels = expanded_labels.astype('uint32')
    elif max_value <= np.iinfo(np.uint64).max:
        expanded_labels = expanded_labels.astype('uint64')

    logger.info(f"Expanded labels data type: {expanded_labels.dtype}")
    io.imsave(fname=os.path.join(output_path), arr=expanded_labels)

def main():
    args = get_args()
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel.upper())
    type_of_input = check_input_outputs(args)
    expand_mask(args.input, args.output, args.pixels, type_of_input=type_of_input)

if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info(f"Execution time: {time.time() - start_time:.1f} seconds ")


"""
Example:
python expand_mask.py \
--log-level "DEBUG" \
--input "cylinter_demo/mask/15.tif" \
--output "cylinter_demo/output/15_expanded.tif" \
--pixels 5
"""