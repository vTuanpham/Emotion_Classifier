import os
import cv2
import sys
sys.path.insert(0, r"./")
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from src.models.classifier import EmotionClassifier
from src.data.dataloader import EmotionDataloader, EMOTIONS


class EmotionPredictor:
    def __init__(self,
                 face_dectect_cpkt_path: str="./src/checkpoints/haarcascade_frontalface_default.xml",
                 emotion_classifier_path: str="./src/checkpoints/best_model.pt",
                 live_cam: bool=False
                 ):
        self.face_dectect_cpkt_path = face_dectect_cpkt_path
        self.emotion_classifier_path = emotion_classifier_path
        self.live_cam = live_cam

    def load_emotion_classifier(self):
        model = EmotionClassifier()
        model.load_state_dict(torch.load(self.emotion_classifier_path, map_location='cpu'), strict=False)
        return model

    def load_face_detect_cpkt(self):
        face_cascade = cv2.CascadeClassifier(self.face_dectect_cpkt_path)
        return face_cascade

    def inference(self):
        val_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        model = self.load_emotion_classifier()
        if self.live_cam:
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = self.load_face_detect_cpkt()
                faces = face_cascade.detectMultiScale(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
                    X = resize_frame / 256
                    X = Image.fromarray((X))
                    X = val_transform(X).unsqueeze(0)
                    with torch.no_grad():
                        model.eval()
                        log_ps = model.cpu()(X)
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        pred = EMOTIONS[int(top_class.numpy())]
                    cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        else:
            model = self.load_emotion_classifier()
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507395516207,), (0.255128989415,))
            ])
            dataloaders = EmotionDataloader(r"./fer_2013",
                                            val_transform=val_transform,
                                            num_worker=1)
            dataloaders = dataloaders.__call__()
            for face, _ in iter(dataloaders['test']):
                # Note: In test data, the label is None

                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_cascade = self.load_face_detect_cpkt()
                faces = face_cascade.detectMultiScale(face)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
                    X = resize_frame / 256
                    X = Image.fromarray((X))
                    X = val_transform(X).unsqueeze(0)
                    with torch.no_grad():
                        model.eval()
                        log_ps = model.cpu()(X)
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        pred = EMOTIONS[int(top_class.numpy())]
                    cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    predictor = EmotionPredictor(live_cam=True)
    predictor.inference()



