import classify_image
import cv2
import glob
import json
import librosa
import numpy as np
import os
import pygame as pg
from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog
from screeninfo import get_monitors
from sentence_transformers import SentenceTransformer
import sys
import webbrowser

mixer = pg.mixer
mixer.init()

def play_audio(file_path):
    mixer.music.load(file_path)
    mixer.music.play()

def pause_audio():
    mixer.music.pause()

def get_screen_size():
    monitors = get_monitors()
    if monitors:
        return (monitors[0].width, monitors[0].height)
    return 800, 600

def get_image_paths(folder_path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    try:
        image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except ValueError:
        print("Warning: Some images could not be sorted numerically.")
    return image_paths

def load_and_scale_images(folder_path, size):
    image_paths = get_image_paths(folder_path)
    images = []
    for img_path in image_paths:
        img = pg.image.load(img_path).convert_alpha()
        img = pg.transform.scale(img, size)
        images.append(img)
    return images, image_paths

def browse_file(file_filter = None):
    app = QApplication(sys.argv)
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(filter=file_filter)
    return file_path if file_path else None

def euclidean_distance(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)

def get_text_from_user():
    app = QApplication(sys.argv)
    text, ok = QInputDialog.getText(None, 'Are you okay?', 'What happened?')
    if ok:
        return text
    return None

def load_json_data(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)


def find_closest_record(user_input, json_data):
    SENTENCE_TRANSFORMER_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    user_emb = SENTENCE_TRANSFORMER_MODEL.encode(user_input)
    min_distance = float('inf')
    closest_record = None
    
    for record in json_data:
        description = record.get('benefits', '')
        description_emb = SENTENCE_TRANSFORMER_MODEL.encode(description)
        distance = euclidean_distance(user_emb, description_emb)
        
        if distance < min_distance:
            min_distance = distance
            closest_record = record
    
    return closest_record


def classify_audio(src_file_path, n_mfcc=13):
    if src_file_path.endswith(".wav"):
        from classify_mfcc import classify_mfcc
        y, sr = librosa.load(src_file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        label = classify_mfcc(mfcc)
        print(type(label))
        if label:
            return label
        return None
    return None

def classify_img(category, image_file_path, num_classes):
    result = classify_image.infer_single_image(category, image_file_path, num_classes)
    return result

def open_site(link):
    webbrowser.open(link)

def show_image(image_path, category, text, font_scale=0.8, thickness=2, background_height=100):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    font_color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    background = np.zeros((background_height, width, 3), dtype=np.uint8)

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (background_height + text_size[1]) // 2
    cv2.putText(background, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    final_image = np.vstack((image, background))

    cv2.imshow(f'{category.title()}: {text.upper()}', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

