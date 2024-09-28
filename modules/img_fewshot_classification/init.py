import cv2 as cv
import os
import pickle
import numpy as np
from scipy.spatial import distance
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import torch
from torchvision import transforms
from modules.img_fewshot_classification import classifier

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = classifier.get_classifier()

def get_embedding_for_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        emb = model(image)
        if emb.dim() > 2:
            emb = emb.view(emb.size(0), -1)
        return emb.cpu().numpy()

class App:
    def __init__(self):
        self.det = HandDetector(detectionCon=0.8, maxHands=1)
        self.color = (0, 0, 0)
        self.innercolor = (255, 255, 255)
        self.CAM = cv.VideoCapture(0)
        if not self.CAM.isOpened():
            print("Error: Camera not accessible.")
            exit(1)
        
        self.frame_width = int(self.CAM.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.CAM.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        self.button_width = 200
        self.button_height = 80
        self.margin = 20
        self.current_class = None
        self.image_count = 0
        self.is_recording = False
        self.input_text = ""
        self.show_input = False
        self.mode = 'capture'
        self.status_text = ""
        self.create_buttons()
        self.embeddings_dict = self.load_embeddings()
        self.cam()

    def load_embeddings(self):
        filepath = 'data/fewshot/tablet/embeddings.pkl'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as file:
                return pickle.load(file)
        else:
            return {}

    def save_embeddings(self):
        filepath = 'data/fewshot/tablet/embeddings.pkl'
        with open(filepath, 'wb') as file:
            pickle.dump(self.embeddings_dict, file)
        print("Embeddings saved to embeddings.pkl")

    def create_buttons(self):
        self.button_rects = {
            "NEW CLASS": (self.frame_width - self.button_width - self.margin, self.margin,
                     self.frame_width - self.margin, self.margin + self.button_height),
            "CANCEL": (self.frame_width - self.button_width - self.margin, 
                       self.margin + self.button_height + self.margin,
                       self.frame_width - self.margin,
                       self.margin + 2 * self.button_height + self.margin),
            "CLASSIFY": (self.frame_width - self.button_width - self.margin, 
                         self.margin + 2 * self.button_height + 2 * self.margin,
                         self.frame_width - self.margin,
                         self.margin + 3 * self.button_height + 2 * self.margin),
            "CLEAR DB": (self.frame_width - self.button_width - self.margin, 
                      self.margin + 3 * self.button_height + 3 * self.margin,
                      self.frame_width - self.margin,
                      self.margin + 4 * self.button_height + 3 * self.margin)
        }

    def cam(self):
        while True:
            status, self.frame = self.CAM.read()
            if not status:
                print("Error: Failed to capture image.")
                break
            self.frame = cv.flip(self.frame, 1)
            hands, self.frame = self.det.findHands(self.frame, flipType=False)

            for label, rect in self.button_rects.items():
                x1, y1, x2, y2 = rect
                cv.rectangle(self.frame, (x1, y1), (x2, y2), (0,0,0), -1)
                cv.putText(self.frame, label, (x1 + 10, y1 + 60), cv.FONT_HERSHEY_DUPLEX, 1, self.innercolor, 2)

            if hands:
                hand = hands[0]
                lmList = hand['lmList']
                x, y = lmList[8][0], lmList[8][1]  # Index finger tip coordinates

                for label, rect in self.button_rects.items():
                    x1, y1, x2, y2 = rect
                    if x1 < x < x2 and y1 < y < y2:
                        if label == "NEW CLASS":
                            self.start_class_entry()
                        elif label == "CANCEL":
                            self.end()
                        elif label == "CLASSIFY":
                            self.switch_to_classifier_mode()
                        elif label == "CLEAR DB":
                            self.clear_embeddings()
                        self.status_text = f"{label}"

            if self.show_input and self.mode == 'capture':
                self.display_input_text()
            if self.is_recording and self.current_class and self.mode == 'capture':
                self.display_progress()

            if self.mode == 'classifier' and hasattr(self, 'nearest_class'):
                self.display_nearest_class()

            self.display_status_text()

            cv.imshow('CLASSIFY SUBTLENESS', self.frame)
            key = cv.waitKey(1)
            if key == 27:
                break
            elif key == ord('c') and self.is_recording and self.current_class and self.mode == 'capture':
                self.display_progress()
            elif key >= 32 and key <= 126 and self.show_input and self.mode == 'capture':
                self.input_text += chr(key)
            elif key == 8 and self.show_input and self.mode == 'capture':
                self.input_text = self.input_text[:-1]
            elif key == 13 and self.show_input and self.mode == 'capture':
                self.add_class()
            elif key == 13 and self.is_recording and self.current_class and self.mode == 'capture':
                self.capture_image()
                self.display_progress()
            elif key == 13 and self.mode == 'classifier':
                self.classify_image()

        self.end()

    def start_class_entry(self):
        self.input_text = ""
        self.show_input = True
        self.current_class = None
        self.is_recording = False
        self.mode = 'capture'
        self.status_text = "Enter class name and press Enter."

    def add_class(self):
        class_name = self.input_text.strip()
        if class_name:
            class_name = ''.join(c for c in class_name if c.isalnum() or c in (' ', '_')).rstrip()
            self.current_class = class_name
            self.image_count = 0
            self.is_recording = True
            self.show_input = False
            self.input_text = ""
            self.status_text = f"Recording images for class: {self.current_class}"
            self.create_buttons()
        else:
            self.status_text = "Class name cannot be empty."

    def capture_image(self):
        if self.image_count >= 10:
            self.is_recording = False
            self.show_class_created_message()
            return

        status, frame = self.CAM.read()
        if not status:
            self.status_text = "Error: Failed to capture image."
            return

        frame = cv.flip(frame, 1)
        temp_file = "temp_image.jpg"
        success = cv.imwrite(temp_file, frame)
        if success:
            self.image_count += 1
            emb = get_embedding_for_image(temp_file)
            self.embeddings_dict[tuple(emb.flatten())] = self.current_class
            self.status_text = f"Image captured for class: {self.current_class}"
            os.remove(temp_file)
        else:
            self.status_text = "Failed to save image."

    def display_input_text(self):
        cv.putText(self.frame, f"CLASS NAME: {self.input_text}", (20, self.frame_height - 50), cv.FONT_HERSHEY_DUPLEX, 1, self.innercolor, 2)

    def display_progress(self):
        progress_text = f"CAPTURED {self.image_count}/10 IMAGES"
        cv.putText(self.frame, progress_text, (20, self.frame_height - 50), cv.FONT_HERSHEY_DUPLEX, 1, self.innercolor, 2)

    def show_class_created_message(self):
        message = "Class Created Successfully!"
        cv.putText(self.frame, message, (20, self.frame_height - 50), cv.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

    def switch_to_classifier_mode(self):
        self.mode = 'classifier'
        self.show_input = False
        self.status_text = "Classifier mode activated."

    def classify_image(self):
        status, frame = self.CAM.read()
        if not status:
            self.status_text = "Error: Failed to capture image."
            return

        frame = cv.flip(frame, 1)
        temp_file = "temp_image.jpg"
        success = cv.imwrite(temp_file, frame)
        if success:
            emb = get_embedding_for_image(temp_file)
            os.remove(temp_file)
            
            if self.embeddings_dict:
                embeddings_list = np.array(list(self.embeddings_dict.keys()))
                distances = distance.cdist([emb.flatten()], embeddings_list, 'euclidean')
                closest_index = np.argmin(distances)
                closest_embedding = embeddings_list[closest_index]
                self.nearest_class = self.embeddings_dict[tuple(closest_embedding)]
                self.status_text = f"{self.nearest_class}"
            else:
                self.status_text = "No embeddings available for classification."
        else:
            self.status_text = "Failed to save image."

    def display_nearest_class(self):
        pass
    '''
        text = f"{self.nearest_class}"
        (w, h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, 1, 2)
        x = (self.frame_width - w) // 2
        y = self.frame_height - 50
        cv.rectangle(self.frame, (x - 10, y - h - 10), (x + w + 10, y + 10), (0, 0, 0), -1)
        cv.putText(self.frame, text, (x, y), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    '''
    def display_status_text(self):
        text = self.status_text
        (w, h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, 1, 2)
        x = self.frame_width - w - 20
        y = self.frame_height - 20
        cv.rectangle(self.frame, (x - 10, y - h - 10), (x + w + 10, y + 10), (0, 0, 0), -1)
        cv.putText(self.frame, text, (x, y), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    def clear_embeddings(self):
        filepath = 'data/fewshot/tablet/embeddings.pkl'
        if os.path.exists(filepath):
            os.remove(filepath)
            self.embeddings_dict = {}
            self.status_text = "Embeddings cleared."
        else:
            self.status_text = "No embeddings file found to clear."

    def end(self):
        self.CAM.release()
        self.save_embeddings()
        cv.destroyAllWindows()

def init():
    App()
