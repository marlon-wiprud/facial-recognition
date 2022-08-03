import cv2
import os
import numpy as np
from PIL import Image
from service.utils import get_faces, maybe_create_dir, get_face_cascade, IMAGE_HEIGHT, IMAGE_WIDTH
import random


def horizontal_flip(img):
    return cv2.flip(img, 1)


def vertical_flip(img):
    return cv2.flip(img, 0)


def zoom(img, value):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


def preview_imgs(imgs):
    show_img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    for img in imgs:
        show_img = np.concatenate((show_img, img), axis=1)

    cv2.imshow('Result', show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def augment_by_path(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return img, augment_face(img.copy())


def augment_face(img):
    zoomed_img = zoom(img.copy(), 0.3)
    h_flipped_img = horizontal_flip(img.copy())
    zomed_h_flipped_img = zoom(h_flipped_img.copy(), 0.3)
    v_flipped_img = vertical_flip(img.copy())
    zomed_v_flipped_img = zoom(v_flipped_img.copy(), 0.3)

    rotate_10 = rotation(img.copy(), 10)
    rotate_20 = rotation(img.copy(), 20)
    rotate_30 = rotation(img.copy(), 30)
    rotate_40 = rotation(img.copy(), 40)

    return [zoomed_img, h_flipped_img, zomed_h_flipped_img, v_flipped_img, zomed_v_flipped_img, rotate_10, rotate_20, rotate_30, rotate_40]


def preprocess_imgs(images):
    face_cascade = get_face_cascade()

    output = []

    for idx, img in enumerate(images):
        print('preprocessing idx ', idx)

        image_array = np.array(img, 'uint8')
        faces = get_faces(img, face_cascade)

        if len(faces) < 1:
            print('no faces, skipping index ', idx)
            continue

        for (x_, y_, w, h) in faces:
            size = (IMAGE_WIDTH, IMAGE_HEIGHT)

            roi = image_array[y_: y_ + h, x_: x_ + w]

            if len(roi) < 1:
                print('no roi, skipping index ', idx)
                continue

            zero_shape = False
            for x in roi.shape:
                if x == 0:
                    zero_shape = True

            if zero_shape == True:
                print('zero shape, skipping ', idx, roi.shape)
                continue

            resized_image = cv2.resize(roi, size)
            image_array = np.array(resized_image, 'uint8')
            output_img = Image.fromarray(image_array)
            output.append(output_img)

    return output


def preprocess_data():

    data_folder = "data"

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

                        size = (IMAGE_WIDTH, IMAGE_HEIGHT)

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
