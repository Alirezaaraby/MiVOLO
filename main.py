from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult

import os

import cv2
import torch

if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

def age_gender_predictor(img):
  try:
    image = cv2.imread(img)

    detector = Detector("./yolov8x_person_face.pt", "cuda:0", verbose=True)

    age_gender_model = MiVOLO(
        "mivolo_imbd.pth.tar",
        "cuda:0",
        half=True,
        use_persons=True,
        disable_faces=False,
        verbose=True,
    )

    detected_objects: PersonAndFaceResult = detector.predict(image)
    age_gender_model.predict(image, detected_objects)

    out_im, age, sex = detected_objects.plot()

    cv2.imwrite("output.jpg",out_im)
  except Exception as e:
    print(e)
print("hi")
age_gender_predictor("1.jpg")