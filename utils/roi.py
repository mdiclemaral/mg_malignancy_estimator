import os
from pathlib import Path
from typing import Self, Tuple

import cv2
import numpy as np
import torch

__all__ = [
    "ROIextraction",
]


class ROIextraction:
    def __init__(
        self,
        device: torch.device,
    ) -> Self:
        """..."""
        self.device = device
        self.H = 416
        self.W = 416

    def load(
        self,
        checkpoint: bytes,
    ) -> None:
        """..."""
        with open(str(Path(__file__).parent / "yolo.pt"), "wb") as f:
            f.write(checkpoint)

        self.net = torch.hub.load(
            str(Path(__file__).parent / "yolov5"),
            "custom",
            source="local",
            path=str(Path(__file__).parent / "yolo.pt"),
            device=self.device,
        )

        os.remove(str(Path(__file__).parent / "yolo.pt"))

    def __call__(
        self,
        pixel_array: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """..."""
        # YOLOv5
        pixel_array, (xmin, ymin, xmax, ymax), success = self.run_method(pixel_array, method="yolo")
        if success:
            return pixel_array, (xmin, ymin, xmax, ymax)

        # Otsu Thresholding
        pixel_array, (xmin, ymin, xmax, ymax), success = self.run_method(pixel_array, method="otsu")
        if success:
            return pixel_array, (xmin, ymin, xmax, ymax)

        return pixel_array, (xmin, ymin, xmax, ymax)

    def run_method(
        self, pixel_array: np.ndarray, method: str = "yolo"
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], bool]:
        """..."""
        H, W = pixel_array.shape[:2]
        success = True

        try:
            # YOLOv5
            if method == "yolo":
                x = cv2.resize((pixel_array.copy() * 255.0).astype(np.uint8), (self.W, self.H))
                detections = self.net(x)
                results = detections.pandas().xyxy[0].to_dict(orient="records")

                if len(results) > 0:
                    xmin, ymin, xmax, ymax = (
                        results[0]["xmin"],
                        results[0]["ymin"],
                        results[0]["xmax"],
                        results[0]["ymax"],
                    )
                    xmin, ymin, xmax, ymax = (
                        int(xmin * W / self.W),
                        int(ymin * H / self.H),
                        int(xmax * W / self.W),
                        int(ymax * H / self.H),
                    )
                else:
                    xmin, ymin, xmax, ymax = 0, 0, W, H
                    success = False
            # Otsu Thresholding
            elif method == "otsu":
                x = cv2.GaussianBlur((pixel_array[..., 0].copy() * 255.0).astype(np.uint8), (5, 5), 0)
                _, x = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(x.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour = max(contours, key=cv2.contourArea)
                xmin, ymin, w, h = cv2.boundingRect(contour)
                xmax, ymax = xmin + w, ymin + h

                if xmax > xmin + 10 and ymax > ymin + 10:
                    pass
                else:
                    xmin, ymin, xmax, ymax = 0, 0, W, H
                    success = False
            else:
                pass
        except:
            xmin, ymin, xmax, ymax = 0, 0, W, H
            success = False

        return pixel_array[ymin:ymax, xmin:xmax], (xmin, ymin, xmax, ymax), success
    