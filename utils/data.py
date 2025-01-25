import cv2
from typing import Dict
import numpy as np


def load_image(image_path) -> np.ndarray:
    """ load image from path """
    return cv2.imread(image_path)

def split_image(image: np.ndarray) -> Dict[str, np.ndarray]:
    """ split image into 4 quadrants and return as list of views"""
    if len(image.shape) == 3:
        h, w, _ = image.shape
    elif len(image.shape) == 2:
        h, w = image.shape
    h2, w2 = h//2, w//2
    return {"L_MLO": image[:h2, :w2], "R_MLO": image[:h2, w2:], "L_CC": image[h2:, :w2], "R_CC": image[h2:, w2:]}
    # return {"L_MLO": image[:h2, w2:], "R_MLO": image[:h2, :w2], "L_CC": image[h2:, w2:], "R_CC": image[h2:, :w2]}

