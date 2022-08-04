from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Model, load_model
from keras_vggface.vggface import VGGFace
import pickle
import cv2
from service.utils import save_training_output, get_face_cascade, IMAGE_HEIGHT, IMAGE_WIDTH, EPOCHS, load_latest_models
from service.preprocessing import preprocess_imgs, save_preprocessed_images
from keras_vggface.utils import preprocess_input
import numpy as np


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

    save_training_output(model, train_generator)

    # save_model(model, SAVE_PATH)
    # save_training_labels(train_generator, CLASS_DICT_SAVE_PATH)


def predict_faces(model, faces, class_list, img):
    image_array = np.array(img, "uint8")

    for (x_, y_, w, h) in faces:
        size = (IMAGE_WIDTH, IMAGE_HEIGHT)
        roi = image_array[y_: y_ + h, x_: x_ + w]
        resized_image = cv2.resize(roi, size)

        # prepare the image for prediction
        predict_img = img_to_array(resized_image)
        predict_img = np.expand_dims(predict_img, axis=0)
        predict_img = preprocess_input(predict_img, version=1)

        # making prediction
        predicted_prob = model.predict(predict_img)
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


def recognize_face(path):

    model, class_list = load_latest_models()

    face_cascade = get_face_cascade()

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    prediction = predict_faces(model, face_cascade, class_list, img)

    return prediction


def cleanup_webcam(stream):
    stream.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


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

    save_preprocessed_images(images, label)


def draw_faces(img, faces):
    for (x, y, w, h) in faces:
        color = (0, 255, 255)  # rgb
        stroke = 5
        cv2.rectangle(img, (x, y), (x + w, y + h), color, stroke)


def extract_faces(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_cascade = get_face_cascade()

    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5)

    return faces


def run_webcam():
    stream = cv2.VideoCapture(0)

    model, class_list = load_latest_models()

    while(True):
        (_, frame) = stream.read()
        faces = extract_faces(frame)
        draw_faces(frame, faces)
        prediction = predict_faces(model, faces, class_list, frame)

        print("PREDICTION: ", prediction)

        # show the frame
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):    # Press q to break out
            break

    cleanup_webcam()
