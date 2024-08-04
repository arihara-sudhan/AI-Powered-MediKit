import cv2
import json
import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean
import torch
from triplet_class import Model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def get_image_embedding(image_path, device=torch.device("cpu")):
    image = Image.open(image_path).convert('RGB')
    image = Model.transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = Model.model(image)
    return embedding

def load_existing_embeddings(json_file_path):
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing embeddings from JSON: {e}")
            return {}
    return {}

def find_nearest(embedding):
    json_file_path = "meta/embeddings.json"
    embeddings_data = load_existing_embeddings(json_file_path)
    if not embeddings_data:
        print("No embeddings found in the JSON file.")
        return None
    
    min_distance = float('inf')
    closest_class = None

    for class_name, class_embeddings in embeddings_data.items():
        for stored_embedding in class_embeddings:
            # Convert embeddings to numpy arrays and ensure they are 1-D
            embedding_np = np.array(embedding).flatten()
            stored_embedding_np = np.array(stored_embedding).flatten()

            # Print shapes for debugging
            print(f"Embedding shape: {embedding_np.shape}")
            print(f"Stored embedding shape: {stored_embedding_np.shape}")

            # Calculate distance
            distance = euclidean(embedding_np, stored_embedding_np)
            if distance < min_distance:
                min_distance = distance
                closest_class = class_name
    return closest_class

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('SHOW THE TABLET AND CLICK ENTER', frame)
            key = cv2.waitKey(1)
            if key == 13:
                cv2.imwrite("./data/frame.jpg", frame)
                emb = get_image_embedding("./data/frame.jpg")
                label = find_nearest(emb)
                if label:
                    print(label)
            elif key == 27:
                break
        else:
            print("Error: Unable to Capture Frame...")
    cap.release()
    cv2.destroyAllWindows()