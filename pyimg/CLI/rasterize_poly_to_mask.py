# basics for CLI
import time
import os
from loguru import logger
import sys
import argparse
from pathlib import Path

# packages for image processing
import spatialdata
import skimage.io

def get_args():
    """Get arguments from command line"""
    description = """Create a projection of the nucleus and membrane channels for cellpose segmentation"""
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
    assert os.path.isdir(args.input), "Input must be a directory"
    assert args.input.endswith(".zarr"), "Input file must be a .zarr folder"
    #output
    assert args.output.endswith(".tif"), "Output file must be a .tif file"
    #log level
    assert args.loglevel in ["DEBUG", "INFO"], "Log level must be either DEBUG or INFO"

def load_image(path_to_zarr):
    """Load .zarr as sdata"""
    sdata = spatialdata.read_zarr(path_to_zarr)
    logger.info(f"Loaded sdata from {path_to_zarr}")
    return sdata

def get_width_height(sdata):
    """Obtain max width and height from sdata"""
    #we assume the reference image is the first image in the list
    ref_image_key = list(sdata.images.keys())[0]
    logger.info(f"Reference image key: {ref_image_key}")
    logger.info(f"Reference image shape: {sdata.images[ref_image_key]['scale0'].image.values.shape}")
    #hardcoding that it is a pyramid with scale0 as base layer with shape cyx
    max_y, max_x = sdata.images[ref_image_key]['scale0'].image.values.shape[1:]
    logger.info(f"Max height: {max_y}, Max width: {max_x},")
    return max_y, max_x

def rasterize(sdata, max_y, max_x):
    """Rasterize sdata image"""
    logger.info(f"Rasterizing sdata")
    sdata["mask"] = spatialdata.rasterize(
        data = sdata["cellpose_boundaries"],
        axes = ["y", "x"],
        min_coordinate = [0, 0],
        max_coordinate = [max_y, max_x],
        target_coordinate_system = "pixels",
        target_unit_to_pixels = 1.0)
    return sdata

def main():
    args = get_args()
    check_inputs_paths(args)
    #logging setup
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel.upper())
    #sdata
    sdata = load_image(args.input)
    max_y, max_x = get_width_height(sdata)
    sdata = rasterize(sdata, max_y, max_x)
    #save mask as tif
    skimage.io.imsave(args.output, sdata.images['mask'].values)
    
if __name__ == "__main__":
    time_start = time.time()
    main()
    logger.info(f"Elapsed time: {int(time.time() - time_start)} seconds")


#example command
"""
python project.py \
--input /Users/jnimoca/Jose_BI/P26_SOPA_seg/small_image.zarr \
--output /Users/jnimoca/Jose_BI/P26_SOPA_seg/mask.tif \
"""