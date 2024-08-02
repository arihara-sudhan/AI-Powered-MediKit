import cv2
import torch
import pickle
from PIL import Image
import os
import json

"""
with open('meta/meta.json') as f:
    data = json.load(f)
"""

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load index and train labels
#with open('./Index/data.pickle', 'rb') as f:
#    index, train_labels = pickle.load(f)

# Function to get image embedding
def get_image_embedding(image_path, device=torch.device("cpu")):
    image = Image.open(image_path).convert('RGB')
    #image = Model.transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        #embedding = Model.model(image)
        pass
#    return embedding

# Function to find nearest neighbor
def find_nearest_neighbor(embedding):
#    _, nearest_index = index.search(embedding.view(1, -1), 1)
#    return train_labels[nearest_index.item()][0]
    pass

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Display the captured frame
            cv2.imshow('SHOW THE TABLET AND CLICK ENTER', frame)

            # Check for key press events
            key = cv2.waitKey(1)
            if key == 13:  # 13 is the ASCII code for the return key
                # Save the current frame as "frame.jpg"
                cv2.imwrite("./data/frame.jpg", frame)
                embedding = get_image_embedding("./data/frame.jpg")
                if embedding:
                    label = find_nearest_neighbor(embedding)
                    print(label)
                    return label
                else:
                    label = "Some Error Occured!"
                #speak(data.get(label))

            elif key == 27:
                break
        else:
            print("Error: Unable to Capture Frame...")
    cap.release()
    cv2.destroyAllWindows()

