from tensorflow.keras.models import load_model
import pickle
from cv2.data import haarcascades
import cv2
import os

DATA_PATH = "./cleaned_data"
SAVE_PATH = "./models"
CLASS_DICT_SAVE_PATH = "face-labels.pickle"
EPOCHS = 15
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


def maybe_create_dir(dir_path):
    isExist = os.path.exists(dir_path)
    if isExist is False:
        os.makedirs(dir_path)


def load_training_labels(file_name):
    with open(file_name, "rb") as \
            f:
        class_dictionary = pickle.load(f)

    return class_dictionary


def get_class_list():
    class_dict = load_training_labels(CLASS_DICT_SAVE_PATH)
    class_list = [value for _, value in class_dict.items()]
    return class_list


def load_local_model(save_path):
    return load_model(save_path + 'test_model.h5')


def get_faces(img, face_cascade):
    return face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)


def get_face_cascade():
    return cv2.CascadeClassifier(haarcascades + 'haarcascade_frontalface_default.xml')
