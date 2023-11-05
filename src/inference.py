import os
import cv2
import sys
import csv

import numpy as np

sys.path.insert(0, r"./")
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.metrics import classification_report

from src.models.classifier import EmotionClassifier
from src.data.dataloader import EmotionDataloader, EMOTIONS


class EmotionPredictor:
    def __init__(self,
                 face_dectect_cpkt_path: str="./src/checkpoints/haarcascade_frontalface_default.xml",
                 emotion_classifier_path: str="./src/checkpoints/best_model.pt",
                 live_cam: bool=False,
                 interactive: bool=True,
                 predict_csv: bool=False,
                 ):
        self.face_dectect_cpkt_path = face_dectect_cpkt_path
        self.emotion_classifier_path = emotion_classifier_path
        self.interactive = interactive
        self.predict_csv = predict_csv
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
            transforms.ToTensor(),
            transforms.Normalize((0.507395516207,), (0.255128989415,))
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

        elif self.interactive:
            model = self.load_emotion_classifier()
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507395516207,), (0.255128989415,))
            ])
            dataloaders = EmotionDataloader(r"./src/data/fer_2013/fer2013/fer2013.csv",
                                            val_transform=val_transform,
                                            num_worker=1, val_batch_size=1)
            dataloaders = dataloaders.__call__()
            fig, axs = plt.subplots(1, 1)  # Create a figure with 3 subplots

            for i, (face, label) in enumerate(iter(dataloaders['test'])):
                with torch.no_grad():
                    model.eval()
                    log_ps = model.cpu()(face)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    pred = EMOTIONS[int(top_class.numpy())]

                # Convert the PyTorch tensor to a NumPy array and squeeze the channel dimension
                face_np = np.asarray(face.squeeze())

                # Display the image with the label on the corresponding subplot
                axs.imshow(face_np)
                axs.set_title(f"Emotion predict: {pred} "
                              f"\nActual predict: {EMOTIONS[int(label.numpy())]} ")

                plt.draw()
                plt.pause(0.001)
                user_input = input("Press 'n' to move to the next example or 'q' to quit: ")
                if user_input == 'q':
                    break

            plt.close()

        elif self.predict_csv:
            model = self.load_emotion_classifier()
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507395516207,), (0.255128989415,))
            ])
            dataloaders = EmotionDataloader(r"./src/data/fer_2013/fer2013/fer2013.csv",
                                            val_transform=val_transform,
                                            num_worker=1, val_batch_size=1)
            dataloaders = dataloaders.__call__()
            preds = []
            for i, (face, _) in enumerate(iter(dataloaders['test'])):
                with torch.no_grad():
                    model.eval()
                    log_ps = model.cpu()(face)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    preds.append(int(top_class.numpy()))

            print("\n Writing predictions to csv...")
            with open("submission.csv", 'w', newline='') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerows([[pred] for pred in preds])

        else:
            model = self.load_emotion_classifier()
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507395516207,), (0.255128989415,))
            ])
            dataloaders = EmotionDataloader(r"./src/data/fer_2013/fer2013/fer2013.csv",
                                            val_transform=val_transform,
                                            num_worker=2, val_batch_size=20)
            dataloaders = dataloaders.__call__()
            accuracy = 0
            true_labels = []
            predicted_labels = []
            for i, (faces, labels) in enumerate(iter(dataloaders['test'])):
                with torch.no_grad():
                    # Forward pass
                    log_ps = model(faces)
                    ps = torch.exp(log_ps)

                    # Get predicted labels
                    _, predicted = torch.max(ps, 1)

                    true_labels.extend(labels.cpu().numpy())
                    predicted_labels.extend(predicted.cpu().numpy())

            print(classification_report(true_labels, predicted_labels))

            # Calculate accuracy
            accuracy = (torch.tensor(predicted_labels) == torch.tensor(true_labels)).sum().item() / len(true_labels)
            print("Accuracy:", accuracy)

            # Calculate precision
            precision = precision_score(true_labels, predicted_labels, average='macro')
            print("Precision:", precision)

            # Calculate F1 score
            f1 = f1_score(true_labels, predicted_labels, average='macro')
            print("F1 Score:", f1)

            # Calculate confusion matrix
            confusion_mat = confusion_matrix(true_labels, predicted_labels)
            print("Confusion Matrix:")
            print(confusion_mat)

            # Plot confusion matrix
            class_names = [emote for emote in EMOTIONS.values()]  # Replace with your actual class names
            fig, ax = plt.subplots()
            im = ax.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(confusion_mat.shape[1]),
                   yticks=np.arange(confusion_mat.shape[0]),
                   xticklabels=class_names, yticklabels=class_names,
                   xlabel='Predicted label', ylabel='True label',
                   title='Confusion Matrix')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            fmt = '.2f' if confusion_mat.dtype == 'float' else 'd'
            thresh = confusion_mat.max() / 2.
            for i in range(confusion_mat.shape[0]):
                for j in range(confusion_mat.shape[1]):
                    ax.text(j, i, format(confusion_mat[i, j], fmt),
                            ha="center", va="center",
                            color="white" if confusion_mat[i, j] > thresh else "black")
            plt.show()
            fig.savefig("Confusion_matrix.png", bbox_inches='tight')

            return accuracy, precision, f1, confusion_mat


if __name__ == "__main__":
    predictor = EmotionPredictor(live_cam=False,
                                 interactive=False,
                                 predict_csv=False)
    predictor.inference()



