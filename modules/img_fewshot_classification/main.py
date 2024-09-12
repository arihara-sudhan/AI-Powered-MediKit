import cv2 as eye
from cvzone.HandTrackingModule import HandDetector as hdari
import time

class App:
    def __init__(self):
        self.det = hdari(detectionCon=0.8, maxHands=1)
        self.color = (255, 255, 255)
        self.innercolor = (0, 0, 0)
        self.CAM = eye.VideoCapture(0)
        self.frame_width = int(self.CAM.get(3))
        self.frame_height = int(self.CAM.get(4))
        self.button_width = 200
        self.button_height = 80
        self.margin = 20
        self.create_buttons()
        self.cam()

    def create_buttons(self):
        self.button_rects = {
            "Create Classes": (self.frame_width - self.button_width - self.margin, self.margin, 
                                self.frame_width - self.margin, self.margin + self.button_height),
            "Classify": (self.frame_width - self.button_width - self.margin, 
                         self.margin + self.button_height + self.margin, 
                         self.frame_width - self.margin, 
                         self.margin + 2 * self.button_height + self.margin)
        }

    def cam(self):
        while True:
            status, self.frame = self.CAM.read()
            self.frame = eye.flip(self.frame, 1)
            hands, self.frame = self.det.findHands(self.frame, flipType=False)
            
            # Draw buttons
            for label, rect in self.button_rects.items():
                x1, y1, x2, y2 = rect
                eye.rectangle(self.frame, (x1, y1), (x2, y2), self.color, -1)
                eye.putText(self.frame, label, (x1 + 10, y1 + 60), eye.FONT_HERSHEY_DUPLEX, 1, self.innercolor, 2)

            if hands:
                hand = hands[0]
                lmList = hand['lmList']
                x, y = lmList[8][0], lmList[8][1]

                for label, rect in self.button_rects.items():
                    x1, y1, x2, y2 = rect
                    if x1 < x < x2 and y1 < y < y2:
                        print(f"Button Pressed: {label}")

            eye.imshow('ARI', self.frame)
            key = eye.waitKey(1)
            if key == 27:  # ESC key
                break

        self.end()

    def end(self):
        self.CAM.release()
        eye.destroyAllWindows()

App()
