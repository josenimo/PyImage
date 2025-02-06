# basics for CLI
import time
import os
from loguru import logger
import sys
import argparse
from pathlib import Path
import tifffile
import dask.array as da
from dask_image import imread
import numpy as np

def get_args():
    """Get arguments from command line"""
    description = """Script to scale down a .tif image to a lower bit depth using Dask."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-i", "--input",        dest="input",       action="store", required=True, help="File path to input image.")
    inputs.add_argument("-o", "--output",       dest="output",      action="store", required=True, help="Path to output image.")
    inputs.add_argument("-ll", "--log-level",   dest="loglevel",    default='INFO', choices=["DEBUG", "INFO"], help='Set the log level (default: INFO)')
    arg = parser.parse_args()
    arg.input = os.path.abspath(arg.input)
    arg.output = os.path.abspath(arg.output)
    return arg

def check_inputs_paths(args):
    """ check inputs and outputs """
    #input
    assert os.path.isfile(args.input), "Input must be a file"
    assert args.input.endswith(".tif"), "Input file must be a .tif file"
    #output
    assert args.output.endswith(".tif"), "Output file must be a .tif file"
    #log level
    assert args.loglevel in ["DEBUG", "INFO"], "Log level must be either DEBUG or INFO"

def metadata_parse(img):
    omexml = tifffile.OmeXml()
    omexml.addimage(
        dtype=img.dtype, shape=img.shape, axes='CYX',
        storedshape=(img.shape[0], 1, 1, img.shape[1], img.shape[2], 1))
    omexml = omexml.tostring()
    return omexml

def scale_channel_and_write(image_path, output_path):
    image = imread.imread(image_path)
    logger.info(f"Image shape: {image.shape} ")
    assert image.ndim == 3, "Image must have 3 dimensions"
    logger.info(f"Current data type : {image.dtype} ")
    logger.info(f"Estimated image size: {image.nbytes / 1e9:.4g} GB")
    omexml = metadata_parse(image)

    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        for channel in range(image.shape[0]):
            logger.info(f"      Processing channel {channel}")
            channel_array = image[channel,:,:]
            channel_array = channel_array.astype(np.float32)
            channel_array = channel_array - channel_array.min().compute()
            channel_array = channel_array / channel_array.max().compute()
            channel_array = channel_array * np.iinfo(np.uint8).max
            channel_array = channel_array.astype(np.uint8)
            logger.info(f"      Done processing channel {channel}")
            tif.write(channel_array.compute(), description=omexml, contiguous=True)
            omexml = None
            logger.info(f"      Channel {channel} written to {output_path}")

def main():
    args = get_args()
    check_inputs_paths(args)
    #logging setup
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel.upper())
    scale_channel_and_write(args.input, args.output)
    
if __name__ == "__main__":
    time_start = time.time()
    main()
    logger.info(f"Elapsed time: {int(time.time() - time_start)} seconds")


#example command
"""
python downscale_image_dask.py \
--input /Users/jnimoca/Jose_BI/P26_SOPA_seg/small_image.tif \
--output /Users/jnimoca/Jose_BI/P26_SOPA_seg/small_image_8bit.tif \
"""