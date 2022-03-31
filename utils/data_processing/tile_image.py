from itertools import product
import cv2


def tile_image(img, patch_size, stride, out_dir, scene):
    heigth, width, channels = img.shape
    for corner in product(
        list(range(0, heigth - patch_size, int(patch_size * stride)))
        + [heigth - patch_size],
        list(range(0, width - patch_size, int(patch_size * stride)))
        + [width - patch_size],
    ):
        left, down = corner
        right, top = left + patch_size, down + patch_size
        crop_img = img[left:right, down:top, :]
        if crop_img.shape == (patch_size, patch_size, channels):
            filename = (
                f'{out_dir}/{scene.replace(".tif", "")}_{left}_{down}_{right}_{top}.tif'
            )
            cv2.imwrite(filename, crop_img)
