import torch
import numpy as np

from typing import Dict, List, Tuple

from utils.roi import ROIextraction
from utils.process import ImageProcessingManager
from utils.mass import MassSegmentation
from utils.process import resize_array, remove_array_padding
import cv2

processing_manager = ImageProcessingManager(
    canvas_shape=(2000, 1600),
    mass_segment_shape=(1024, 768),
    mass_segment_color=(28, 252, 88)
)


def remove_redundant_area(roi_extraction: ROIextraction,
    original_pixel_arrays: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, int, int, int]]]:
    """..."""
    pixel_arrays = {}
    boundaries = {}

    for key, pixel_array in original_pixel_arrays.items():
        pixel_array, boundary = roi_extraction(pixel_array.copy())
        pixel_arrays[key] = pixel_array
        boundaries[key] = boundary

    return pixel_arrays, boundaries


def preprocess(
    pixel_arrays: Dict[str, np.ndarray],
) -> torch.Tensor | List[torch.Tensor]:
    """..."""
    input_pixel_arrays = {"mass_segment": []}
    # Preprocessing for Segmenta
    for key, pixel_array in pixel_arrays.items():
        pixel_array_mass_segment = processing_manager.preprocess_segmenta(
            pixel_array=pixel_array,
            key=key,
        )
        input_pixel_arrays["mass_segment"].append(pixel_array_mass_segment)  # LMLO, RMLO, LCC, RCC
    input_pixel_arrays["mass_segment"] = torch.cat(input_pixel_arrays["mass_segment"], dim=0)

    
    # input_pixel_arrays["mass_segment"] = torch.cat(input_pixel_arrays["mass_segment"], dim=0)

    return input_pixel_arrays["mass_segment"]


def inference(
    mass_model: MassSegmentation,
    input_pixel_arrays: Dict[str, torch.Tensor | List[torch.Tensor]],
    keys: List[str]
) -> Dict[str, Dict[str, np.ndarray]]:
    """..."""
    model_outputs = {}

    pixel_arrays_mass_segment = input_pixel_arrays
    masks_mass_segment = mass_model(pixel_arrays_mass_segment)
    masks_mass_segment = {key: mask for key, mask in zip(keys, masks_mass_segment)}

    model_outputs["mass_segment"] = masks_mass_segment

    return model_outputs["mass_segment"]


def postprocess(idd,
    original_pixel_arrays: Dict[str, np.ndarray],
    model_outputs: Dict[str, Dict[str, np.ndarray]],
    boundaries: Dict[str, Tuple[int, int, int, int]],
    keys: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Dict[str, bool]]]:
    """..."""
    output_pixel_arrays = {}
    for key in keys:
        iddd= f"{idd}_{key}"
        pixel_array = original_pixel_arrays[key]
        boundary = boundaries[key]

        mask_mass_segment = model_outputs[key]  

        pixel_array, output_pixel_array= processing_manager.postprocess_segmenta(iddd,
            pixel_array=pixel_array,
            key=key,
            mask_mass_segment=mask_mass_segment,
            boundary=boundary,
        )

        original_pixel_arrays[key] = pixel_array
        output_pixel_arrays[key] = output_pixel_array

    return original_pixel_arrays, output_pixel_arrays