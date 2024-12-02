import geopandas as gpd
import argparse
import os
from loguru import logger
import dask_image.imread
from shapely.geometry import shape, MultiPolygon
from rasterio.features import shapes

def get_args():
    parser = argparse.ArgumentParser(description='Convert .tif mask to geojson polygons')
    parser.add_argument('--input', dest="input", type=str, help='Path to input .tif file')
    parser.add_argument('--output', dest="output", type=str, help='Path to output geojson file')
    args = parser.parse_args()
    args.input = os.path.abspath(args.input)
    return args

def check_inputs_paths(args):
    assert os.path.isfile(args.input), "Input file does not exist"
    assert args.input.endswith((".tif", ".tiff")), "Input file must be a .tif or .tiff file"
    assert args.output.endswith(".geojson"), "Output file must be a .geojson file"

def load_mask(input_path):
    mask = dask_image.imread.imread(input_path)
    logger.info(f"Loaded mask with shape {mask.shape}, dtype {mask.dtype}, max {mask.max().compute()}, min {mask.min().compute()}")
    return mask

def process_mask(input_path):
    """
    Converts a labeled segmentation mask into a GeoDataFrame.
    """
    mask = load_mask(input_path)
    if mask.ndim > 2:
        mask = mask[0]  # Assuming single-channel mask (take the first channel if multi-channel)
    
    # Compute the mask as a numpy array, this is then memory limited
    mask_np = mask.compute()

    # Dictionary to store geometries grouped by cell ID
    cell_geometries = {}

    # Extract shapes and corresponding values
    for shape_dict, cell_id in shapes(mask_np, mask=(mask_np > 0)):
        polygon = shape(shape_dict)
        cell_id = int(cell_id)
        if cell_id not in cell_geometries:
            cell_geometries[cell_id] = []
        cell_geometries[cell_id].append(polygon)

    # Combine multiple polygons into MultiPolygons if needed
    cell_ids = []
    geometries = []
    for cell_id, polygons in cell_geometries.items():
        if len(polygons) == 1:
            geometries.append(polygons[0])  # Single Polygon
        else:
            geometries.append(MultiPolygon(polygons))  # Combine into MultiPolygon
        cell_ids.append(cell_id)

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({'cellId': cell_ids, 'geometry': geometries}, crs="EPSG:4326")
    return gdf

def save_geojson(gdf, output_path):
    gdf.to_file(output_path, driver="GeoJSON")
    logger.info(f"GeoJSON saved to {output_path}")

def main():
    args = get_args()
    check_inputs_paths(args)
    gdf = process_mask(args.input)
    save_geojson(gdf, args.output)
    print(f"Success")

if __name__ == "__main__":
    main()