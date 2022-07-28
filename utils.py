import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from keras_vggface import utils
import pickle
from cv2.data import haarcascades
import cv2

DATA_PATH = "./cleaned_data"
SAVE_PATH = "./models"
CLASS_DICT_SAVE_PATH = "face-labels.pickle"
EPOCHS = 20
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


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


def predict_faces(model, face_cascade, class_list, img):
    image_array = np.array(img, "uint8")
    faces = get_faces(img, face_cascade)
    for (x_, y_, w, h) in faces:
        size = (IMAGE_WIDTH, IMAGE_HEIGHT)
        roi = image_array[y_: y_ + h, x_: x_ + w]
        resized_image = cv2.resize(roi, size)

        # prepare the image for prediction
        predict_img = image.img_to_array(resized_image)
        predict_img = np.expand_dims(predict_img, axis=0)
        predict_img = utils.preprocess_input(predict_img, version=1)

        # making prediction
        predicted_prob = model.predict(predict_img)
        print('prediction 1 =>', predicted_prob[0].argmax())
        probability = predicted_prob[0].argmax()
        prediction = {
            'probability':  probability,
            'prediction': class_list[probability],
            'coordinates': {
                'x': x_,
                'y': y_,
                'w': w,
                'h': h,
            }
        }

        return prediction


def test():
    img = cv2.imread("./data/marlon/IMG-3064.jpg", cv2.IMREAD_COLOR)

    class_list = get_class_list()

    model = load_local_model(SAVE_PATH)

    face_cascade = get_face_cascade()

    predict_faces(model, face_cascade, class_list, img)


# test()
