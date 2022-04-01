import rasterio
import numpy as np


class Tiff:
    def __init__(self) -> None:
        pass
        
    @staticmethod
    def process_raster(raster_path: str) -> np.array:
        if 'WV03' in raster_path:
            with rasterio.open(raster_path) as src:
                img = src.read([1])
                
        else:
            raise NotImplementedError(f'Sensor not supported for {raster_path}')
        width = src.width
        height = src.height
        transforms = src.transform
        meta = src.meta
        return img, width, height, transforms, meta
