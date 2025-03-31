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

#TODO it seems that the processing loads the whole image everytime instead of using the previous processed layer
# processing is actually 90% of processing time, so it can increase time efficiency greatly, especially for large images

def get_args():
    """Get arguments from command line"""
    description = """Script to scale down a .tif image to a lower bit depth using Dask."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-i", "--input",        dest="input",       action="store", required=True, help="File path to input image or folder with images")
    inputs.add_argument("-o", "--output",       dest="output",      action="store", required=True, help="Path to output image or folder.")
    inputs.add_argument("-t", "--tile-size",    dest="tile_size",   action="store", type=int, default=1072, help="Tile size for pyramid generation (must be divisible by 16)")
    inputs.add_argument("-c", "--compress",     dest="compress",    action="store", default=True, choices=[True, False], help="compresses image bit-depth to 8bit, anything that is not True does not")
    inputs.add_argument("-ll", "--log-level",   dest="loglevel",    action="store", default='INFO', choices=["DEBUG", "INFO"], help='Set the log level (default: INFO)')

    arg = parser.parse_args()
    arg.input = os.path.abspath(arg.input)
    arg.output = os.path.abspath(arg.output)
    return arg

def check_inputs_paths(args):
    """ check inputs and outputs """

    #TILE SIZE
    assert isinstance(args.tile_size,int), "Tile size must be integer"
    assert args.tile_size % 16 == 0, "Tile size must be divisible by 16"
    #LOG LEVEL
    assert args.loglevel in ["DEBUG", "INFO"], "Log level must be either DEBUG or INFO"
    #COMPRESS
    assert args.compress == True or args.compress == False, "Must be True or False"
    # INPUT
    if os.path.isfile(args.input):
        assert args.input.lower().endswith((".tif", ".tiff")), "File must be a .tif or .tiff"
        assert args.output.lower().endswith((".tif", ".tiff")), "Output file must end in .tif or .tiff"
        return "single_file"
    elif os.path.isdir(args.input):
        assert os.path.isdir(args.output), "Input is a directory, but output is not"
        try:
            os.makedirs(args.output, exist_ok=True)
        except OSError:
            raise ValueError('Could not create output folder, check permissions')
        return "folder"

def setup_logger(log_level):
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=log_level)

def metadata_parse(img):
    omexml = tifffile.OmeXml()
    omexml.addimage(
        dtype=img.dtype, shape=img.shape, axes='CYX',
        storedshape=(img.shape[0], 1, 1, img.shape[1], img.shape[2], 1))
    omexml = omexml.tostring()
    return omexml

def check_image(image_path):
    image = imread.imread(image_path)
    logger.info(f"Image shape: {image.shape} ")
    logger.info(f"Current data type : {image.dtype} ")
    logger.info(f"Estimated image size: {image.nbytes / 1e9:.4g} GB")
    assert image.ndim == 3, "Image must have 3 dimensions"

def scale_channel_and_write(image_path, output_path, tile_size=1072, compress_8bit=True):
    image = imread.imread(image_path)
    omexml = metadata_parse(image)

    # Calculate the number of subifds
    subifds = (np.ceil(np.log2(max(1, max(image.shape[-2:]) / tile_size))) + 1).astype(int)

    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        
        for channel in range(image.shape[0]):
            
            channel_array = image[channel,:,:]

            if compress_8bit:
                logger.info(f"  Processing channel {channel}")
                channel_array = channel_array.astype(np.float32)
                channel_array = channel_array - channel_array.min().compute()
                channel_array = channel_array / channel_array.max().compute()
                channel_array = channel_array * np.iinfo(np.uint8).max
                channel_array = channel_array.astype(np.uint8)
                logger.info(f"  Done processing channel {channel}")
            
            tif.write(
                channel_array.compute(), 
                description=omexml,
                metadata=False,
                subifds=subifds, 
                tile=(tile_size, tile_size),
                photometric='minisblack',
                contiguous=True)
            omexml = None
            logger.info(f"  Channel {channel} baselayer written to {output_path}")

            for level in range(subifds):
                logger.info(f"      Processing level {level} for channel {channel}")
                res = 2**(level+1)
                tif.write(
                    channel_array[::res, ::res].compute(),
                    subfiletype=1,
                    metadata=False,
                    tile=(tile_size, tile_size),
                    photometric='minisblack')
                logger.info(f"      Done writing level {level} for channel {channel}")







#CHATGPT

def scale_channel_and_write(image_path, output_path, tile_size=1072, compress_8bit=True):
    image = imread.imread(image_path)
    omexml = metadata_parse(image)

    # Calculate the number of subifds
    subifds = (np.ceil(np.log2(max(1, max(image.shape[-2:]) / tile_size))) + 1).astype(int)

    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        
        for channel in range(image.shape[0]):
            
            channel_array = image[channel, :, :]

            if compress_8bit:
                logger.info(f"  Processing channel {channel}")
                channel_array = channel_array.astype(np.float32)
                channel_array = (channel_array - channel_array.min().compute()) / (channel_array.max().compute())
                channel_array *= np.iinfo(np.uint8).max
                channel_array = channel_array.astype(np.uint8)
                logger.info(f"  Done processing channel {channel}")
            
            # Compute the base layer
            base_layer = channel_array.compute()
            tif.write(
                base_layer, 
                description=omexml,
                metadata=False,
                subifds=subifds, 
                tile=(tile_size, tile_size),
                photometric='minisblack',
                contiguous=True)
            omexml = None
            logger.info(f"  Channel {channel} baselayer written to {output_path}")

            # Downsample progressively
            downsampled = base_layer
            for level in range(subifds):
                logger.info(f"      Processing level {level} for channel {channel}")

                downsampled = downsampled[::2, ::2]  # Use previously computed downsampled image
                tif.write(
                    downsampled,
                    subfiletype=1,
                    metadata=False,
                    tile=(tile_size, tile_size),
                    photometric='minisblack')

                logger.info(f"      Done writing level {level} for channel {channel}")

#CHATGPT








def main():
    args = get_args()
    file_or_folder = check_inputs_paths(args)
    setup_logger(args.loglevel.upper())
    
    if file_or_folder == "single_file":
        check_image(args.input)
        scale_channel_and_write(args.input, args.output, tile_size=args.tile_size, compress_8bit=args.compress)

    elif file_or_folder == "folder":
        list_of_files = [file for file in os.listdir(args.input) if file.lower().endswith((".tif", ".tiff"))]
        for file in list_of_files:
            check_image(os.path.join(args.input, file))
        for file in list_of_files:
            scale_channel_and_write(
                image_path=os.path.join(args.input, file),
                output_path=os.path.join(args.output, file),
                tile_size=args.tile_size,
                compress_8bit=args.compress)

    
if __name__ == "__main__":
    time_start = time.time()
    main()
    logger.info(f"Elapsed time: {int(time.time() - time_start)} seconds")


#example command
"""
python downscale_image_dask.py \
--input /Users/jnimoca/Jose_BI/P26_SOPA_seg/small_image.tif \
--output /Users/jnimoca/Jose_BI/P26_SOPA_seg/small_image_8bit.tif \

https://forum.image.sc/t/writing-contiguous-ome-tiff-with-tifffile/70613
"""
