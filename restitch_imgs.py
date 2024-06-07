import zarr
import numpy as np

from cellpose.dynamics import compute_masks
from skimage import io


def reconstruct_image_from_zarr(zarr_fp):
    # Load the Zarr store
    zarr_store = zarr.open(zarr_fp, mode='r')

    # Retrieve the original image dimensions and other attributes
    original_image_dimensions = zarr_store.attrs['original_image_dimensions']
    overlap = zarr_store.attrs['overlap']
    tile_size = zarr_store.attrs['tile_size']

    # Initialize an empty array with the original image dimensions
    flows = np.zeros((2, *original_image_dimensions[1:]))
    probs = np.zeros(original_image_dimensions[1:])

    # Initialize a count array to keep track of how many tiles contribute to each pixel (for averaging)
    count = np.zeros(original_image_dimensions[1:])

    # Iterate over the tiles in the Zarr store
    for tile in zarr_store['flow_fields']:

        # get the masks and find the flow field and prob map for the tile

        # NOT this: ...
        # Load the tile
        flow_tile = zarr_store['flow_fields'][tile][:]
        prob_tile = zarr_store['prob_maps'][tile][:]

        # permute the flow_tile if the channels are in the last dimension
        if flow_tile.shape[2] == 2:
            flow_tile = np.transpose(flow_tile, (2, 0, 1))

        # Calculate the position of the tile in the original image
        # The tile name is in the format 'tile_x{X}_y{Y}'
        x_i = int(tile.split('_')[1][1:])
        y_i = int(tile.split('_')[2][1:])
        x = x_i * (tile_size[0] - overlap)
        y = y_i * (tile_size[1] - overlap)

        # Place the tile in the correct position in the array
        flows[:, x:x+flow_tile.shape[1], y:y+flow_tile.shape[2]] += flow_tile
        probs[x:x+prob_tile.shape[0], y:y+prob_tile.shape[1]] += prob_tile

        # Update the count array
        count[x:x+flow_tile.shape[1], y:y+flow_tile.shape[2]] += 1

    # Average the overlapping regions, only divide by count where count is non-zero
    count[count == 0] = 1
    flows /= count
    probs /= count

    # Run the mask reconstruction
    masks, _ = compute_masks(flows, probs)

    return masks, flows, probs

if __name__ == '__main__':
    zarr_fp = '/media/willow/BigBoi/ivan_imgs/original_images/CG8_Merged.zarr'
    masks, flows, probs = reconstruct_image_from_zarr(zarr_fp)

    # save masks, flows, and probs to the zarr store
    zarr_store = zarr.open(zarr_fp, mode='a')
    zarr_store['stitched_masks'] = masks
    zarr_store['stitched_flows'] = flows
    zarr_store['stitched_probs'] = probs

    io.imsave('masks.tif', masks)
