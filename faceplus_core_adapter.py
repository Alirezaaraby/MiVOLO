from core_adapter import CoreAdapter
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
import glob

class FacePlusCoreAdapter(CoreAdapter):
    def __init__(self) -> None:
        super(FacePlusCoreAdapter, self).__init__("faceplus")
        self.loaded = False

    def load(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.known_embeddings = self.load_known_faces()
        self.unknown_embeddings = self.load_unknown_faces()
        self.loaded = True

    def face_detection(self, frame):
        img_rgb = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        boxes, faces = self.mtcnn.detect(img_rgb, landmarks=False)
        return img_rgb, boxes, faces

    def process_frame(self, frame, id, index=0, save=False, threshold=0.6):
        img_rgb, boxes, faces = self.face_detection(frame)
        predictions = []

        if faces is not None:
            for box in boxes:
                x, y, x2, y2 = [int(coord) for coord in box]
                face_crop = img_rgb[y:y2, x:x2]
                face = self.transform(face_crop).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    face_embedding = self.resnet(face)

                similarities = {
                    name: torch.nn.functional.cosine_similarity(known_emb, face_embedding).item()
                    for name, known_emb in self.known_embeddings.items()
                }

                best_match, best_score = max(similarities.items(), key=lambda x: x[1], default=("Unknown", 0))
                label = best_match if best_score > threshold else "Unknown"

                if label == 'Unknown': 
                    unknown_faces_dir = "./data/unknown"
                    last_unknown_id = self.get_last_unknown_face(unknown_faces_dir)

                    unknown_similarities = {
                        unknown_id: torch.nn.functional.cosine_similarity(unknown_emb, face_embedding).item()
                        for unknown_id, unknown_emb in self.unknown_embeddings.items()
                    }

                    best_unknown_match, best_unknown_score = max(unknown_similarities.items(), key=lambda x: x[1], default=(None, 0))

                    if best_unknown_score > threshold:
                        label = f"Unknown_{best_unknown_match}"  # Return matched unknown ID
                    else:
                        new_unknown_id = int(last_unknown_id) + 1
                        self.unknown_embeddings[new_unknown_id] = face_embedding
                        label = f"Unknown_{new_unknown_id}"
                        unknown_path = os.path.join(unknown_faces_dir, f"unknown_{new_unknown_id}.jpg")
                        cv2.imwrite(unknown_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                else:
                    face_crop = None
                predictions.append((label, face_crop))

        return predictions
    
    def load_known_faces(self):
        known_faces_dir = "./data/known"
        known_embeddings = {}

        for file in os.listdir(known_faces_dir):
            name, ext = os.path.splitext(file)
            if ext.lower() in [".jpg", ".jpeg", ".png"]:
                img = cv2.imread(os.path.join(known_faces_dir, file))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face, prob = self.mtcnn(img_rgb, return_prob=True)

                if face is not None and prob > 0.90:
                    with torch.no_grad():
                        embedding = self.resnet(face.unsqueeze(0).to(self.device))
                    known_embeddings[name] = embedding
        
        return known_embeddings
    
    def load_unknown_faces(self):
        unknown_faces_dir = "./data/unknown"
        unknown_embeddings = {}

        if not os.path.exists(unknown_faces_dir):
            os.makedirs(unknown_faces_dir)

        for file in os.listdir(unknown_faces_dir):
            name, ext = os.path.splitext(file)
            if ext.lower() in [".jpg", ".jpeg", ".png"]:
                img = cv2.imread(os.path.join(unknown_faces_dir, file))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face, prob = self.mtcnn(img_rgb, return_prob=True)

                if face is not None and prob > 0.90:
                    with torch.no_grad():
                        embedding = self.resnet(face.unsqueeze(0).to(self.device))
                    unknown_embeddings[int(name.split("_")[-1])] = embedding  # Assuming file is named 'unknown_1.jpg'
        
        return unknown_embeddings
    

    def get_last_unknown_face(self, unknown_directory):
        list_of_files = glob.glob(os.path.join(unknown_directory, "unknown_*.jpg"))

        if not list_of_files:
            return 0

        latest_file = max(list_of_files, key=os.path.getctime)
        filename = os.path.basename(latest_file)

        try:
            return int(filename.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            return 0