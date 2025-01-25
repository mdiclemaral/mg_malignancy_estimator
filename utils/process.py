from typing import Dict, Self, Tuple, List, Any

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as tvF
import matplotlib.pyplot as plt


__all__ = [
    "ImageProcessingManager",
]


# Methods
def resize_tensor(
    pixel_array: torch.Tensor,
    shape: Tuple[int, int],
) -> torch.Tensor:
    """..."""
    H, W = pixel_array.shape[1:]

    ratioH, ratioW = H / shape[0], W / shape[1]
    if ratioH > ratioW:
        target_shape = (shape[0], int(W / ratioH))
    else:
        target_shape = (int(H / ratioW), shape[1])

    return tvF.resize(pixel_array, target_shape).unsqueeze(0)


def tensor_padding(
    pixel_array: torch.Tensor,
    shape: Tuple[int, int],
    key: str,
) -> torch.Tensor:
    """..."""
    H, W = pixel_array.shape[2:]
    tensor_padded = torch.zeros((1, 3) + shape).to(pixel_array.dtype)

    if key[0] == "R":
        tensor_padded[..., :H, -1 * W :] = pixel_array
    else:
        tensor_padded[..., :H, :W] = pixel_array

    return tensor_padded


def resize_array(
    pixel_array: np.ndarray,
    shape: Tuple[int, int],
    force: bool = False,
) -> np.ndarray:
    """..."""
    if force:
        return cv2.resize(pixel_array, shape[::-1])

    H, W = pixel_array.shape[:2]

    ratioH, ratioW = H / shape[0], W / shape[1]
    if ratioH > ratioW:
        target_shape = (shape[0], int(W / ratioH))
    else:
        target_shape = (int(H / ratioW), shape[1])

    return cv2.resize(pixel_array, target_shape[::-1])


def array_padding(
    pixel_array: np.ndarray,
    shape: Tuple[int, int],
    key: str,
) -> np.ndarray:
    """..."""
    H, W = pixel_array.shape[:2]
    array_padded = np.zeros(shape + (3,), dtype=pixel_array.dtype)

    if key[0] == "R":
        array_padded[:H, -1 * W :, ...] = pixel_array
    else:
        array_padded[:H, :W, ...] = pixel_array

    return array_padded


def remove_array_padding(
    pixel_array: np.ndarray,
    boundary: Tuple[int, int, int, int],
    key: str,
) -> np.ndarray:
    """..."""
    H, W = pixel_array.shape[:2]
    bH, bW = boundary[3] - boundary[1], boundary[2] - boundary[0]

    ratioH, ratioW = H / bH, W / bW
    if ratioH > ratioW:
        H, W = int(ratioW * bH), int(ratioW * bW)
    else:
        H, W = int(ratioH * bH), int(ratioH * bW)

    if key[0] == "R":
        pixel_array = pixel_array[:H, -1 * W :]
    else:
        pixel_array = pixel_array[:H, :W]
    return pixel_array


def get_text_position(
    center: Tuple[int, int],
    img_center: Tuple[int, int],
    roi: Tuple[int, int, int, int],
    probability: float
    ) -> Tuple[int, int]:
    """
    center, img_center: center of mass and image respectively
    roi: mass coordinates for text placement
    probability: malignancy probability to calculate length of text to be written
    Returns: text_position in (x, y) format
    """
    center_x, center_y = center
    img_center_x, img_center_y = img_center
    x, y, w, h = roi

    text_size = cv2.getTextSize(f"{probability:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]  # adjust position wrt text_size
    if center_x < img_center_x and center_y < img_center_y: # mass in top left: text to bottom right
        text_position = (x + w + 10, y + h + 20)
    elif center_x >= img_center_x and center_y < img_center_y:  # mass in top right: text to bottom left
        text_position = (x - text_size, y + h + 20)
    elif center_x < img_center_x and center_y >= img_center_y:  # mass in bottom left: text to top right
        text_position = (x + w + 10, y - 10)
    else:  # mass in bottom right: text to top left
        text_position = (x - text_size, y - 10)
        
    return text_position

def find_boundaries(mask, probabilities: int):
    """
    Returns: x, y, w, h bounding box that surrounds probability value
    """
    coords_list = []
    for probability in probabilities:
        coords = np.argwhere(mask == probability)
        if coords.size == 0:
            continue
        leftmost = coords[:, 1].min()
        rightmost = coords[:, 1].max()
        topmost = coords[:, 0].min()
        bottommost = coords[:, 0].max()
        coords_list.append((leftmost, topmost, rightmost - leftmost, bottommost - topmost))
        
    return coords_list


def draw_probability(output_array: np.ndarray, 
                     coords_list: List[Tuple[int, int, int, int]],
                     probabilities: List[int], 
                     boundary: Tuple[int, int, int, int]):
    for (probability, coords) in zip(probabilities, coords_list):
        x, y, w, h = coords
        x, y = x + boundary[0], y + boundary[1]  # offset by the amount of padding
        
        center_x, center_y = x + w // 2, y + h // 2
        img_center_x, img_center_y = output_array.shape[1] // 2, output_array.shape[0] // 2
        text_position = get_text_position((center_x, center_y), (img_center_x, img_center_y), (x, y, w, h), probability)
            
        thickness = 4
        color = (0, 0, 255)
        probability = probability if probability > 0.5 else 1 - probability
        cv2.rectangle(output_array, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(output_array, f"{probability:.2f}", text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)


def extract_contours(output_pixel_array, probability_mask):
    _, binary_mask = cv2.threshold(probability_mask, 0.5, 1, cv2.THRESH_BINARY)
    try:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        return -1
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_pixel_array, (x, y), (x+w, y+h), (255, 0, 0), 2)

        max_prob = np.max(probability_mask[y:y+h, x:x+w])
        cv2.putText(output_pixel_array, f'{max_prob:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# Classes
class ImageProcessingManager:
    def __init__(
        self,
        canvas_shape: Tuple[int, int],
        mass_segment_shape: Tuple[int, int],
        mass_segment_color: Tuple[int, int, int],
    ) -> Self:
        """..."""
        self.i = 0
        self.canvas_shape = canvas_shape
        self.mass_segment_shape = mass_segment_shape
        self.mass_segment_color = np.array(mass_segment_color, dtype=np.uint8)

    def preprocess_segmenta(
        self,
        pixel_array: np.ndarray,
        key: str,
    ) -> torch.Tensor:
        """..."""

        pixel_array = tvF.to_tensor(pixel_array)

        pixel_array_mass_segment = resize_tensor(pixel_array.clone(), self.mass_segment_shape)
        pixel_array_mass_segment = tensor_padding(pixel_array_mass_segment, self.mass_segment_shape, key)

        return pixel_array_mass_segment

    def postprocess_segmenta(
        self,idd,
        pixel_array: np.ndarray | None,
        key: str,
        mask_mass_segment: np.ndarray | None,
        boundary: Tuple[int, int, int, int] | None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, bool]]:
        """..."""
        if pixel_array is None or mask_mass_segment is None:
            return (
                np.zeros(self.canvas_shape + (3,), dtype=np.uint8),  # pixel_array
                np.zeros(self.canvas_shape + (3,), dtype=np.uint8)  # output_pixel_array
            )

        pixel_array = (pixel_array * 255.0).astype(np.uint8)
        output_pixel_array = pixel_array.copy()


        H, W = boundary[3] - boundary[1], boundary[2] - boundary[0]

        output_mass_segment = np.zeros_like(pixel_array)
        # mask_mass_segment = (mask_mass_segment > 0).astype(np.uint8) -> do not need for now ?
        if mask_mass_segment.sum() >= 0:
            probabilities = np.unique(mask_mass_segment)[1:]
            temp_mask = (mask_mass_segment * 255).astype(np.uint8)
            
            temp_mask = remove_array_padding(temp_mask, boundary, key)
            temp_mask = resize_array(temp_mask, (H, W), force=True)
            
            output_mass_segment[boundary[1] : boundary[3], boundary[0] : boundary[2], :] = np.stack([temp_mask] * 3, axis=-1)
            
            normalized_mass_segment = output_mass_segment
            cmap = cv2.applyColorMap((255 - normalized_mass_segment), cv2.COLORMAP_AUTUMN)
            # cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
            
            output_colorized = np.full_like(output_mass_segment, cmap, dtype=np.uint8)
            output_colorized = cv2.addWeighted(output_pixel_array, 0.5, output_colorized, 0.5, 0)
            output_pixel_array = np.where(output_mass_segment > 0, output_colorized, output_pixel_array)
            
            
            # if sum(probabilities) > 0:
            #     print(idd,"probs: ", probabilities)
            # extract_contours(output_pixel_array, output_mass_segment / 255.)
            # coords_list = find_boundaries(temp_mask, uniques)  # find mass boundaries
            # draw_probability(output_pixel_array, coords_list, probabilities, boundary)
            
            
        pixel_array = resize_array(pixel_array, self.canvas_shape)
        pixel_array = array_padding(pixel_array, self.canvas_shape, key)

        output_pixel_array = resize_array(output_pixel_array, self.canvas_shape)
        output_pixel_array = array_padding(output_pixel_array, self.canvas_shape, key)

        # cv2.imwrite(f"output/{key}_pixel_array.png", output_mass_segment)
        # cv2.imwrite(f"output/{key}_colorized.png", output_pixel_array_kk)

        return pixel_array, output_pixel_array
    
    def cat(
        self,
        original_pixel_arrays: Dict[str, np.ndarray],
        output_pixel_arrays: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """..."""
        original_pixel_array = np.concatenate(
            [
                np.concatenate(
                    [original_pixel_arrays["R-MLO"], original_pixel_arrays["L-MLO"]],
                    axis=1,
                ),
                np.concatenate(
                    [original_pixel_arrays["R-CC"], original_pixel_arrays["L-CC"]],
                    axis=1,
                ),
            ],
            axis=0,
        )

        output_pixel_array = np.concatenate(
            [
                np.concatenate(
                    [output_pixel_arrays["R-MLO"], output_pixel_arrays["L-MLO"]],
                    axis=1,
                ),
                np.concatenate(
                    [output_pixel_arrays["R-CC"], output_pixel_arrays["L-CC"]],
                    axis=1,
                ),
            ],
            axis=0,
        )

        return original_pixel_array, output_pixel_array
