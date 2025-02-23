import cv2
import numpy as np
from typing import Union, Dict, Optional, List
import torch
import PIL
from copy import deepcopy


# Dummy detection structure mimicking a YOLO detection object.
class DummyDetection:
    def __init__(
        self,
        bbox: List[float],
        cls: int,
        conf: float = 1.0,
        det_id: Optional[int] = None,
        orig_shape: Optional[List[int]] = None,
    ):
        self.xyxy = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0)  # shape (1,4)
        self.cls = cls
        self.conf = conf
        self.id = torch.tensor(det_id) if det_id is not None else None
        # Expect orig_shape to be [height, width]
        self.orig_shape = (
            orig_shape
            if orig_shape is not None
            else [bbox[3] - bbox[1], bbox[2] - bbox[0]]
        )


# Dummy Results object to mimic the YOLO results expected by PersonAndFaceResult.
class DummyResults:
    def __init__(self, image: np.ndarray, bbox: List[float], category: str):
        # Create a names dictionary mapping class indices to category names.
        # Ensure both "person" and "face" exist.
        # For our dummy result, we set the provided category at index 0 and add "person" if necessary.
        if category == "face":
            self.names = {0: "face", 1: "person"}
            detection_cls = 0
        else:
            self.names = {0: "person", 1: "face"}
            detection_cls = 0

        # For our dummy detection, we consider the bbox to be the whole image.
        h, w = image.shape[:2]
        # If bbox is empty, set to entire image; otherwise use provided bbox.
        if not bbox:
            bbox = [0, 0, w, h]
        self.boxes = [
            DummyDetection(
                bbox, detection_cls, conf=1.0, det_id=None, orig_shape=[h, w]
            )
        ]

        # Store the original image for plotting (if needed)
        self.orig_img = deepcopy(image)
        self.probs = None  # Add this line to initialize the probs attribute

    def plot(self):
        # This dummy plot just returns the original image
        return self.orig_img


# Modified Detector class that uses a dummy results object.
class Detector:
    def __init__(self):
        """Initialize the detector without loading YOLO weights."""
        # You can set up any required default values here.
        pass

    def predict(self, image: Union[np.ndarray, str, "PIL.Image"]):
        """
        Instead of running YOLO, this function creates a dummy results object using the cropped face image.
        """
        if isinstance(image, str):  # If image is a file path
            image = cv2.imread(image)
        elif isinstance(image, PIL.Image.Image):
            image = np.array(image)

        if image is None:
            raise ValueError(
                "Invalid image input. Please ensure the file path is correct or the image format is supported."
            )

        h, w = image.shape[:2]
        bbox = [0, 0, w, h]

        dummy_results = DummyResults(image, bbox, category="face")

        from mivolo.structures import (
            PersonAndFaceResult,
        )

        result = PersonAndFaceResult(dummy_results)
        return result
