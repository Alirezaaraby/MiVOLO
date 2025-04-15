# Test With Single Class

# import cv2
# from faceplus_core_adapter_with_age_gender import FacePlusCoreAdapter

# # Initialize the face recognition adapter
# face_adapter = FacePlusCoreAdapter()
# face_adapter.load()  # Load the model and known faces

# # Load an image frame (example from a file)
# image_path = "image.png"
# frame = cv2.imread(image_path)

# # Process the frame
# result = face_adapter.process_frame(frame, id=1)

# # Print the result
# print(result)

# Test With Seperated Class

import cv2
from faceplus_core_adapter import FacePlusCoreAdapter
from age_gender_core_adapter import MiVOLOCoreAdapter
import numpy as np
from PIL import Image  # Import Image from PIL

# Initialize the face recognition adapter
face_adapter = FacePlusCoreAdapter()
face_adapter.load()  # Load the model and known faces

# Initialize the age detection adapter
age_adapter = MiVOLOCoreAdapter()
age_adapter.load()  # Load the age detection model

# Load an image frame (example from a file)
src = "amir.jpg"
frame = Image.open(src).convert('RGB')

# Process the frame to get face crops and labels
results = face_adapter.process_frame(frame, id=1)

final = []

for label, face_crop in results:
    if face_crop is not None and isinstance(face_crop, np.ndarray) and face_crop.size > 0:
        age, gender = age_adapter.process_frame(face_crop, id=1)
        final.append((label, age, gender))
    else:
        final.append((label, None, None))

print(final)