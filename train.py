import os
import pandas as pd
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras_vggface.vggface import VGGFace

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    './cleaned_data',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

train_generator.class_indices.values()

NO_CLASSES = len(train_generator.class_indices.values())

print("no classes: ", NO_CLASSES)

base_model = VGGFace(include_top=True,
                     model='vgg16',
                     input_shape=(224, 224, 3))
base_model.summary()

print(len(base_model.layers))

model = base_model.output
model = GlobalAveragePooling2D()(model)

model = Dense(1024, activation='relu')(model)
model = Dense(1024, activation='relu')(model)
model = Dense(512, activation='relu')(model)

# final layer with softmax activation
preds = Dense(NO_CLASSES, activation='softmax')(model)

print("model length: ", len(model.layers))

# # don't train the first 19 layers - 0..18
# for layer in model.layers[:19]:
#     layer.trainable = False

# # train the rest of the layers - 19 onwards
# for layer in model.layers[19:]:
#     layer.trainable = True
