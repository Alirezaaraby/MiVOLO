import cv2
import torch
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import PersonAndFaceResult

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

def age_gender_predictor(cropped_face_img_path):
    image = cv2.imread(cropped_face_img_path)

    detector = Detector()  # Now using our dummy detector

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
    print(age, sex)
    cv2.imwrite(cropped_face_img_path.split(".")[0] + "output.jpg", out_im)

print("hi")
age_gender_predictor("face_1.jpg")
