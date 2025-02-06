import dask.array as da
from scipy.ndimage import zoom
from math import ceil

def downscale_dask(dask_array, factors=(1, 0.5, 0.5)):

    assert isinstance(dask_array, da.Array), "Input must be a dask array."
    assert len(factors) != len(dask_array.shape), "The length of `factors` must match the number of dimensions in `dask_array`."

    def rescale_block(block, factors):
        return zoom(block, factors, order=3)  # Cubic interpolation
    
    new_shape = tuple(ceil(s * f) for s, f in zip(dask_array.shape, factors))
    new_chunks = tuple(min(ceil(c * f), ns) for c, f, ns in zip(dask_array.chunksize, factors, new_shape))
    
    return dask_array.map_blocks(
        rescale_block,
        factors=factors,
        dtype=dask_array.dtype,
        chunks=new_chunks
    )