import os
import pandas as pd
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

import pickle
from cv2.data import haarcascades
import cv2

DATA_PATH = "./cleaned_data"
SAVE_PATH = "./models"
CLASS_DICT_SAVE_PATH = "face-labels.pickle"
EPOCHS = 20


def new_train_generator(data_path):
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return train_datagen.flow_from_directory(
        data_path,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical',
        shuffle=True)


def build_model(n_classes):

    base_model = VGGFace(include_top=False,
                         model='vgg16',
                         input_shape=(224, 224, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    # final layer with softmax activation
    preds = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=base_model.inputs, outputs=preds)

    # don't train the first 19 layers - 0..18
    for layer in model.layers[:19]:
        layer.trainable = False

    # train the rest of the layers - 19 onwards
    for layer in model.layers[19:]:
        layer.trainable = True

    return model


def save_model(model, save_path):
    model.save(save_path + "test_model.h5")


def save_training_labels(train_generator, save_path):
    class_dictionary = {
        value: key for key, value in train_generator.class_indices.items()
    }
    with open(save_path, 'wb') as f:
        pickle.dump(class_dictionary, f)
    print(class_dictionary)


def load_training_labels(file_name):
    with open(file_name, "rb") as \
            f:
        class_dictionary = pickle.load(f)

    return class_dictionary


def load_local_model(save_path):
    return load_model(save_path + 'test_model.h5')


def train():
    train_generator = new_train_generator("./cleaned_data")
    n_classes = len(train_generator.class_indices.values())
    model = build_model(n_classes)

    model.summary()

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_generator,
              batch_size=1,
              verbose=1,
              epochs=EPOCHS)

    save_model(model, SAVE_PATH)
    save_training_labels(train_generator, CLASS_DICT_SAVE_PATH)


def test():
    image_width = 224
    image_height = 224

    class_dict = load_training_labels(CLASS_DICT_SAVE_PATH)
    class_list = [value for _, value in class_dict.items()]

    model = load_local_model(SAVE_PATH)

    face_cascade = cv2.CascadeClassifier(
        haarcascades + 'haarcascade_frontalface_default.xml')

    imgtest = cv2.imread("./data/marlon/IMG-3064.jpg", cv2.IMREAD_COLOR)
    image_array = np.array(imgtest, "uint8")

    # get the faces detected in the image
    faces = face_cascade.detectMultiScale(imgtest,
                                          scaleFactor=1.1, minNeighbors=5)
    print(faces)

    for (x_, y_, w, h) in faces:
        # draw the face detected
        # face_detect = cv2.rectangle(
        #     imgtest, (x_, y_), (x_+w, y_+h), (255, 0, 255), 2)
        # # plt.imshow(face_detect)
        # # plt.show()

        # resize the detected face to 224x224
        size = (image_width, image_height)
        roi = image_array[y_: y_ + h, x_: x_ + w]
        resized_image = cv2.resize(roi, size)

        # prepare the image for prediction
        x = image.img_to_array(resized_image)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)

        # making prediction
        predicted_prob = model.predict(x)
        print(predicted_prob)
        print(predicted_prob[0].argmax())
        print("Predicted face: " + class_list[predicted_prob[0].argmax()])
        print("============================\n")


# train()
test()
