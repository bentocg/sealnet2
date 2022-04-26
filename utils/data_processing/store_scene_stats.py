import os

import affine
import pandas as pd


def store_scene_stats(
    scene: str, width: int, height: int, transform: affine.Affine, out_path: str
) -> None:
    """
    Stores scene statistics, later used for output mosaicing

    :param scene: scene name
    :param width: in pixels
    :param height: in pixels
    :param transform: affine matrix from rasterio.open
    :param out_path: path to output .csv file
    """

    # Check if output file exists
    if os.path.exists(out_path):

        # Append to output file
        with open(out_path, "a") as file:
            file.write(
                ",".join(
                    [
                        str(ele)
                        for ele in [scene, width, height]
                        + [
                            transform.a,
                            transform.b,
                            transform.c,
                            transform.d,
                            transform.e,
                            transform.f,
                        ]
                    ]
                )
                + "\n"
            )
    else:

        # Create a new csv file
        stats = pd.DataFrame(
            {
                "scene": scene,
                "width": width,
                "height": height,
                "transform_a": transform.a,
                "transform_b": transform.b,
                "transform_c": transform.c,
                "transform_d": transform.d,
                "transform_e": transform.e,
                "transform_f": transform.f,
            },
            index=[0],
        )
        stats.to_csv(out_path, index=False)
