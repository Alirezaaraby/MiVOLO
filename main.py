import cv2
import torch
from core_adapter import CoreAdapter
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import PersonAndFaceResult
import numpy as np
import os

class MiVOLOCoreAdapter(CoreAdapter):
    
    def __init__(self) -> None:
        super().__init__("mivolo")
        self.loaded = False

    def load(self):
        """Load the age-gender predictor model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = Detector()  # Initialize object detector

        self.age_gender_model = MiVOLO(
            "mivolo_imbd.pth.tar",
            self.device,
            half=True,
            use_persons=True,
            disable_faces=False,
            verbose=True,
        )
        self.loaded = True

    def process_frame(self, frame, id, index=0, save=False):
        """Process an input image frame and predict age and gender."""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert PIL image to OpenCV format if needed
        if isinstance(frame, np.ndarray):  
            image = frame
        else:
            image = np.array(frame)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        detected_objects: PersonAndFaceResult = self.detector.predict(image)
        self.age_gender_model.predict(image, detected_objects)

        out_im, age, sex = detected_objects.plot()
        
        if save:
            self.save_frame(out_im, id, index, age, sex)

        return age, sex

    def save_frame(self, image, id, index, age, sex):
        """Save processed frame with age-gender predictions."""
        dir_path = f'./static/output/{id}/'
        os.makedirs(dir_path, exist_ok=True)

        # Add age & gender text
        cv2.putText(image, f"Age: {age}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Sex: {sex}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Save image
        cv2.imwrite(os.path.join(dir_path, f"mivolo_{index}.jpg"), image)


# Example usage
if __name__ == "__main__":
    adapter = MiVOLOCoreAdapter()
    adapter.load()
    print("Adapter Loaded")

    # Run on a sample image
    img_path = "face_1.jpg"
    image = cv2.imread(img_path)

    age, sex = adapter.process_frame(image, id="test_run", index=1, save=True)
    print("Predicted Age:", age)
    print("Predicted Sex:", sex)
