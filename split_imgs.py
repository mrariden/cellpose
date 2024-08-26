from pathlib import Path
import numpy as np
import zarr
from skimage import io
from tqdm import tqdm
import sys

from cellpose.transforms import normalize_img

from get_image_file_resolution import get_resolution


def extract_tile(image, start_x, start_y, width, height):
    # Function to extract a tile given starting x, y coordinates
    # Ensure that the slicing does not go out of bounds
    end_x = min(start_x + width, image.shape[1])
    end_y = min(start_y + height, image.shape[0])
    return image[start_x:end_x, start_y:end_y]

def process_image_to_zarr(large_image_path, tile_size=1120, overlap=10, zarr_mode='w', skip_intensity_threshold=0.05):
    """
    Processes a large image, splitting it into tiles and storing in a Zarr format.
    The image is normalized using `cellpose.transforms.normalize_img()`
    
    Parameters:
    - large_image_path: Path to the large image.
    - tile_size (int): The width and height of the tiles (assuming square tiles).
    - overlap (int): The number of pixels to overlap tiles.
    - zarr_mode (str): The mode for opening the Zarr store ('w' for write, 'a' for append, etc.).
    
    Returns:
    - A pathlib.Path to the created Zarr store.
    """

    large_image = io.imread(large_image_path)
    large_image = normalize_img(large_image)
    H, W, C = large_image.shape
    
    assert C < W and C < H, f"Image read incorrectly, channel ({C=}) axis larger than height ({H=}) or width ({W=}) axis"

    step_width = step_height = tile_size - overlap

    zarr_fp = Path(large_image_path).with_suffix('.zarr')
    zarr_store = zarr.open(zarr_fp, mode=zarr_mode)
    
    zarr_store.attrs['original_image_dimensions'] = large_image.shape
    zarr_store.attrs['overlap'] = overlap
    zarr_store.attrs['tile_size'] = (tile_size, tile_size)
    zarr_store.attrs['resolution'] = get_resolution(large_image_path)
    

    total_iterations = len(range(0, W, step_width)) * len(range(0, H, step_height))
    
    with tqdm(total=total_iterations, desc='Tiling image into zarr store...') as pbar:
        for x_i, x in enumerate(range(0, W, step_width)):
            for y_i, y in enumerate(range(0, H, step_height)):
                # Adjust the last tile size if it goes beyond the image dimensions
                current_tile_width = min(tile_size, W - x)
                current_tile_height = min(tile_size, H - y)

                tile = extract_tile(large_image, x, y, current_tile_width, current_tile_height)
                
                if tile.size == 0:
                    pbar.update(1)
                    continue

                tile_path = f'tiles/tile_x{x_i}_y{y_i}'
                flow_field_path = f'flow_fields/tile_x{x_i}_y{y_i}'
                prob_map_path = f'prob_maps/tile_x{x_i}_y{y_i}'

                zarr_store[tile_path] = tile
                
                has_some_signal = np.max(tile) > skip_intensity_threshold
                if not has_some_signal:
                    pbar.update(1)
                    continue
                
                if flow_field_path not in zarr_store:
                    zarr_store.create_dataset(flow_field_path, shape=(2, current_tile_width, current_tile_height), dtype=np.float32, fill_value=0)
                if prob_map_path not in zarr_store:
                    zarr_store.create_dataset(prob_map_path, shape=(current_tile_width, current_tile_height), dtype=np.float32, fill_value=0)
                    
                pbar.update(1)

    return zarr_fp


if __name__ == '__main__':
    img_file = sys.argv[1] 
    img_file_zarr_fp = process_image_to_zarr(img_file, tile_size=1000, overlap=100)

