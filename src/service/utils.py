from tensorflow.keras.models import load_model
import pickle
from cv2.data import haarcascades
import cv2
import os
from datetime import datetime
from pathlib import Path

CLEAN_DATA_PATH = "./cleaned_data"
MODEL_FILE_NAME = "model.h5"
CLASS_DICT_FILE_NAME = "face-labels.pickle"
EPOCHS = 10
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


def get_class_list(folder_path):
    class_dict = load_training_labels(
        os.path.join(folder_path, CLASS_DICT_FILE_NAME))
    class_list = [value for _, value in class_dict.items()]
    return class_list


def load_local_model(folder_path):
    return load_model(os.path.join(folder_path, MODEL_FILE_NAME))


def get_latest_model_folder():
    paths = sorted(Path("models").iterdir(), key=os.path.getmtime)
    print(paths)
    if paths[len(paths) - 1]:
        folder = paths[len(paths) - 1]
        return os.path.join("models", folder.name)


def load_latest_models():
    folder_path = get_latest_model_folder()
    model = load_local_model(folder_path)
    class_list = get_class_list(folder_path)
    return model, class_list


def get_faces(img, face_cascade):
    return face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)


def get_face_cascade():
    return cv2.CascadeClassifier(haarcascades + 'haarcascade_frontalface_default.xml')


def new_models_folder():
    folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    relative_path = os.path.join("models", folder_name)
    dir = os.path.join(os.getcwd(), relative_path)
    os.makedirs(dir)
    print('created new model directory: ', dir)
    return relative_path


def save_training_output(model, train_generator):
    dir_path = new_models_folder()

    model.save(os.path.join(dir_path, MODEL_FILE_NAME))

    class_dictionary = {
        value: key for key, value in train_generator.class_indices.items()
    }

    with open(os.path.join(dir_path, CLASS_DICT_FILE_NAME), 'wb') as f:
        pickle.dump(class_dictionary, f)
    print(class_dictionary)
