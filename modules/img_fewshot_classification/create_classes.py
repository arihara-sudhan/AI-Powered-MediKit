import cv2 as cv
import os
from cvzone.HandTrackingModule import HandDetector

class CreateClassesApp:
    def __init__(self):
        self.det = HandDetector(detectionCon=0.8, maxHands=1)
        self.color = (255, 255, 255)
        self.innercolor = (0, 0, 0)
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
        self.create_buttons()
        self.cam()

    def create_buttons(self):
        self.button_rects = {
            "Plus": (self.frame_width - self.button_width - self.margin, self.margin,
                     self.frame_width - self.margin, self.margin + self.button_height),
            "Cancel": (self.frame_width - self.button_width - self.margin, 
                       self.margin + self.button_height + self.margin,
                       self.frame_width - self.margin,
                       self.margin + 2 * self.button_height + self.margin)
        }

    def cam(self):
        while True:
            status, self.frame = self.CAM.read()
            if not status:
                print("Error: Failed to capture image.")
                break
            self.frame = cv.flip(self.frame, 1)
            hands, self.frame = self.det.findHands(self.frame, flipType=False)

            # Draw buttons
            for label, rect in self.button_rects.items():
                x1, y1, x2, y2 = rect
                cv.rectangle(self.frame, (x1, y1), (x2, y2), self.color, -1)
                cv.putText(self.frame, label, (x1 + 10, y1 + 60), cv.FONT_HERSHEY_DUPLEX, 1, self.innercolor, 2)

            if hands:
                hand = hands[0]
                lmList = hand['lmList']
                x, y = lmList[8][0], lmList[8][1]  # Index finger tip coordinates

                for label, rect in self.button_rects.items():
                    x1, y1, x2, y2 = rect
                    if x1 < x < x2 and y1 < y < y2:
                        if label == "Plus":
                            self.start_class_entry()
                        elif label == "Cancel":
                            self.end()

            if self.show_input:
                self.display_input_text()
            if self.is_recording and self.current_class:
                self.display_progress()

            cv.imshow('Create Classes', self.frame)
            key = cv.waitKey(1)
            if key == 27:
                break
            elif key == ord('c') and self.is_recording and self.current_class:
                print(f"Captured {self.image_count}/10 images for class: {self.current_class}")
                self.display_progress()
            elif key >= 32 and key <= 126 and self.show_input:
                self.input_text += chr(key)
            elif key == 8 and self.show_input:
                self.input_text = self.input_text[:-1]
            elif key == 13 and self.show_input:
                self.add_class()
            elif key == 13 and self.is_recording and self.current_class:
                self.capture_image()
                print(f"Captured {self.image_count}/10 images for class: {self.current_class}")
                self.display_progress()

        self.end()

    def start_class_entry(self):
        self.input_text = ""
        self.show_input = True
        self.current_class = None
        self.is_recording = False
        print("Enter class name in the CV window and press Enter.")

    def add_class(self):
        class_name = self.input_text.strip()
        if class_name:
            # Sanitize class name
            class_name = ''.join(c for c in class_name if c.isalnum() or c in (' ', '_')).rstrip()
            # Ensure the 'classes' directory exists
            classes_dir = "classes"
            if not os.path.exists(classes_dir):
                os.makedirs(classes_dir)
            class_dir = os.path.join(classes_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            self.current_class = class_dir
            self.image_count = 0
            self.is_recording = True
            self.show_input = False
            self.input_text = ""
            print(f"Recording images for class: {self.current_class}")
        else:
            print("Class name cannot be empty. Please enter a valid class name.")

    def capture_image(self):
        if self.image_count >= 10:
            self.is_recording = False
            self.show_class_created_message()
            return

        status, frame = self.CAM.read()
        if not status:
            print("Error: Failed to capture image.")
            return

        frame = cv.flip(frame, 1)
        file_name = f"./{self.current_class}/{self.image_count + 1}.jpg"
        success = cv.imwrite(file_name, frame)
        if success:
            self.image_count += 1
            print(f"Image saved: {file_name}")
        else:
            print(f"Failed to save image: {file_name}")

    def display_input_text(self):
        cv.putText(self.frame, f"Enter Class Name: {self.input_text}", (20, self.frame_height - 50), cv.FONT_HERSHEY_DUPLEX, 1, self.innercolor, 2)

    def display_progress(self):
        progress_text = f"Captured {self.image_count}/10 images"
        cv.putText(self.frame, progress_text, (20, self.frame_height - 50), cv.FONT_HERSHEY_DUPLEX, 1, self.innercolor, 2)

    def show_class_created_message(self):
        message = "Class Created Successfully!"
        cv.putText(self.frame, message, (20, self.frame_height-50), cv.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
        cv.imshow('Create Classes', self.frame)
        cv.waitKey(2000)
        self.create_buttons()

    def end(self):
        self.CAM.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    CreateClassesApp()
