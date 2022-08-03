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
from service.utils import predict_faces, get_class_list, get_face_cascade, maybe_create_dir
from service.data_collection import cleanup_webcam
from service.preprocessing import preprocess_imgs

DATA_PATH = "./cleaned_data"
SAVE_PATH = "./models"
CLASS_DICT_SAVE_PATH = "face-labels.pickle"
EPOCHS = 12


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


def recognize_face(path):

    class_list = get_class_list()

    model = load_local_model(SAVE_PATH)

    face_cascade = get_face_cascade()

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    prediction = predict_faces(model, face_cascade, class_list, img)

    return prediction


def introduce_webcam(label):
    stream = cv2.VideoCapture(0)

    frames = []
    count = 0
    run = True

    while run:
        (_, frame) = stream.read()

        if count % 2 == 0:
            frames.append(frame)

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or len(frames) > 50:
            run = False

    cleanup_webcam(stream=stream)

    images = preprocess_imgs(frames)

    cleaned_sub_dir_path = os.path.join('cleaned_data', label)

    maybe_create_dir(cleaned_sub_dir_path)

    for idx, img in enumerate(images):
        save_path = os.path.join(cleaned_sub_dir_path,
                                 label + "_" + str(idx) + '.jpeg')
        print('saving: ', save_path)
        img.save(save_path)
