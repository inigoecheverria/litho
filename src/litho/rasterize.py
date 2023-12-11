try:
    import rasterio
    from rasterio.transform import from_bounds
    rasterio_imported = True
except ModuleNotFoundError:
    rasterio_imported = False
from litho.utils import makedir_from_filename

def rasterize(
    array,
    bounds=[-80,-45,-60,-10], # west,south,east,nort
    crs='EPSG:4326',
    nodata=0,
    filename='raster',
):
    if rasterio_imported is False:
        return
    makedir_from_filename(filename)
    width = array.shape[1]
    height = array.shape[0]
    transform = from_bounds(*bounds,width,height)
    filename = filename + '.tif'
    new_raster = rasterio.open(
        filename, 'w', driver='GTiff',
        height=height, width=width,
        count=1, dtype=str(array.dtype),
        crs=crs, transform=transform, nodata=nodata
    )
    new_raster.write(array, 1)
    new_raster.close()
