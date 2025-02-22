import cv2
from mivolo.predictor import Predictor  # Assuming Predictor is in mivolo.predictor module

# Load the detected face image
face_image = cv2.imread("1.jpg")

# Initialize the Predictor class
class Config:
    def __init__(self):
        self.checkpoint = "mivolo_imbd.pth.tar"
        self.with_persons = True
        self.detector_weights = "./yolov8x_person_face.pt"
        self.device = "cuda:0"
        self.draw = True
        self.disable_faces = False  # Add this line to fix the AttributeError

config = Config()
predictor = Predictor(config, verbose=True)

# Recognize age and gender from the detected face image
detected_objects, out_im, age, sex = predictor.recognize(face_image)
print(sex, age)
# Optionally, save or display the output image with annotations
if out_im is not None:
    cv2.imwrite("output_annotated.jpg", out_im)
    cv2.imshow("Annotated Image", out_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()