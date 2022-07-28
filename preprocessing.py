import cv2
import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from cv2.data import haarcascades


def detect_faces(face_cascade):
    return face_cascade.detectMultiScale(imgtest,
                                         scaleFactor=1.1, minNeighbors=5)


data_folder = "data"
# dimension of images
target_image_width = 224
target_image_height = 224

# for detecting faces
face_cascade = cv2.CascadeClassifier(
    haarcascades + 'haarcascade_frontalface_default.xml')
image_dir = os.path.join(".", data_folder)

current_id = 0
label_ids = {}

for root, _, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):
            path = os.path.join(root, file)
            print('path => ', path)
            label = os.path.basename(root).replace(" ", ".").lower()

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
            image_array = np.array(imgtest, 'uint8')

            faces = detect_faces(face_cascade)

            print('faces => ', len(faces))

            if len(faces) != 1:
                print('photo skipped...')

            # os.remove(path)
            for (x_, y_, w, h) in faces:
                face_detect = cv2.rectangle(
                    imgtest, (x_, y_), (x_ + w, y_ + h), (255, 0, 255), 2)
                # plt.imshow(face_detect)
                # plt.show()

                size = (target_image_width, target_image_height)

                roi = image_array[y_: y_ + h, x_: x_ + w]

                print("roi => ", roi)
                if len(roi) < 1:
                    continue

                resized_image = cv2.resize(roi, size)
                image_array = np.array(resized_image, 'uint8')
                im = Image.fromarray(image_array)
                im.save(os.path.join("cleaned_data/marlon/", file))
