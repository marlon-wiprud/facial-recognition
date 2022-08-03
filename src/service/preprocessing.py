import cv2
import os
import numpy as np
from PIL import Image
from service.utils import get_faces, maybe_create_dir, get_face_cascade


def preprocess_data():

    data_folder = "data"
    # dimension of images
    target_image_width = 224
    target_image_height = 224

    # for detecting faces
    face_cascade = get_face_cascade()

    image_dir = os.path.join(".", data_folder)

    current_id = 0
    label_ids = {}

    for sub_dir in os.listdir(image_dir):
        sub_dir_path = os.path.join(image_dir, sub_dir)
        cleaned_sub_dir_path = os.path.join('cleaned_data', sub_dir)

        maybe_create_dir(cleaned_sub_dir_path)

        for root, _, files in os.walk(sub_dir_path):
            for file in files:
                if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):

                    path = os.path.join(root, file)
                    label = os.path.basename(root).replace(" ", ".").lower()

                    if not label in label_ids:
                        label_ids[label] = current_id
                        current_id += 1

                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    image_array = np.array(img, 'uint8')

                    faces = get_faces(img, face_cascade)

                    if len(faces) < 1:
                        print('no faces, skipping => ', path)
                        continue

                    # os.remove(path)
                    for (x_, y_, w, h) in faces:
                        # face_detect = cv2.rectangle(
                        #     img, (x_, y_), (x_ + w, y_ + h), (255, 0, 255), 2)

                        size = (target_image_width, target_image_height)

                        roi = image_array[y_: y_ + h, x_: x_ + w]

                        if len(roi) < 1:
                            print('no roi, skipping => ', path)
                            continue

                        zero_shape = False
                        for x in roi.shape:
                            if x == 0:
                                zero_shape = True

                        if zero_shape == True:
                            print('zero shape, skipping', path, roi.shape)
                            continue

                        print('resizing: ', path)

                        resized_image = cv2.resize(roi, size)
                        image_array = np.array(resized_image, 'uint8')
                        im = Image.fromarray(image_array)
                        im.save(os.path.join(cleaned_sub_dir_path, file))
