# basics for CLI
import time
import os
from loguru import logger
import sys
import argparse
from pathlib import Path

# packages for image processing
import tifffile
import skimage.io
from ome_types import from_tiff

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

def scale_bit_depth(image_path, max_value=255, numpy_target_dtype:str ='uint8'):
    """Scales the bit depth of an image using Dask."""
    # Load the image using dask-image
    image = skimage.io.imread(image_path)
    logger.info(f"Image shape: {image.shape} ")
    logger.info(f"Current data type : {image.dtype} ")
    logger.info(f"Estimated image size: {image.nbytes / 1e9:.4g} GB")
    # Scale the image to the new bit depth
    scaled_image = (image - image.min()) / (image.max() - image.min()) * max_value
    scaled_image = scaled_image.astype(numpy_target_dtype)
    logger.info(f"Scaled image shape: {scaled_image.shape}")
    logger.info(f"Scaled image data type: {scaled_image.dtype}")
    return scaled_image

def get_metadata(img_path):
    """Detect pixel size from metadata and save it for the new image."""
    ome_metadata = from_tiff(img_path)
    try:
        pixel_size = ome_metadata.images[0].pixels.physical_size_x
    except Exception:
        logger.info("No pixel size found in metadata, default is 1")
        pixel_size = 1.0
    
    metadata = {
        "Pixels": {
            "PhysicalSizeX": pixel_size,
            "PhysicalSizeXUnit": "\u00b5m",
            "PhysicalSizeY": pixel_size,
            "PhysicalSizeYUnit": "\u00b5m",}}
    return metadata

def save_image(image, metadata, output_path):
    """Save the image to the output path."""
    with tifffile.TiffWriter(output_path) as tif:
        tif.write(image, metadata=metadata)
    logger.info(f"Image saved to {output_path}")

def main():
    args = get_args()
    check_inputs_paths(args)
    #logging setup
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel.upper())
    #process image
    image = scale_bit_depth(image_path=args.input, max_value=255, numpy_target_dtype='uint8')
    metadata = get_metadata(args.input)
    save_image(image, metadata, args.output)
    
if __name__ == "__main__":
    time_start = time.time()
    main()
    logger.info(f"Elapsed time: {int(time.time() - time_start)} seconds")


#example command
"""
python project.py \
--input /Users/jnimoca/Jose_BI/P26_SOPA_seg/small_image.tif \
--output /Users/jnimoca/Jose_BI/P26_SOPA_seg/small_image_8bit.tif \
"""