# Import modules
import numpy as np
import matplotlib.pyplot as plt
import time
import os

import tensorflow as tf
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

from multiprocessing import cpu_count

# Record start time for the whole module
start = time.time()

# create generator
datagen = ImageDataGenerator(rescale=1. / 255)

# Prepare dataset iterators
path_data = "../Dataset_mini5b"
path_models = "../models"

dim_xy = 64  #a:28
target_size = (dim_xy,dim_xy)
batch_size = 32  #a:64
seed = 42
interpolation = 'bicubic'
color_mode='grayscale'
class_mode='categorical'

# File names, paths, etc.
path_train = os.path.join(path_data, "train")
path_val = os.path.join(path_data, "val")
path_test1 = os.path.join(path_data, "test1")
path_test2 = os.path.join(path_data, "test2")

train_it = datagen.flow_from_directory(path_train, class_mode=class_mode, target_size=target_size, color_mode=color_mode, batch_size=batch_size, seed=seed)
val_it = datagen.flow_from_directory(path_val, class_mode=class_mode, target_size=target_size, color_mode=color_mode, batch_size=batch_size, seed=seed)
test1_it = datagen.flow_from_directory(path_test1, class_mode=class_mode, target_size=target_size, color_mode=color_mode, batch_size=batch_size, seed=seed)
test2_it = datagen.flow_from_directory(path_test2, class_mode=class_mode, target_size=target_size, color_mode=color_mode, batch_size=batch_size, seed=seed)

# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
#plt.imshow(batchX[2,:,:,:].reshape(28,28), cmap='binary')

def create_model():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=7, activation='relu', padding='same', input_shape=(dim_xy, dim_xy, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))
    model.add(Conv2D(32, kernel_size=5, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))
    model.add(Conv2D(32, kernel_size=5, activation='relu', padding='same'))
    model.add(BatchNormalization())

    #3.
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))  #a:512
    model.add(Dense(1024, activation='relu')) #a:1024

    #4.
    model.add(Dense(5, activation='softmax'))

    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return (model)

def train_model(model):
    workers = cpu_count()

    model_start = time.time()
    history = model.fit_generator(
                                generator=train_it,
                                validation_data=val_it,
                                epochs=3,
                                max_queue_size=32,
                                #workers=workers,
                                #steps_per_epoch=16,    # testing only! leave commented out
                                #validation_steps=16,    # testing only! leave commented out
                                #use_multiprocessing = True,
                                #callbacks=[tb_callback],
                                )
    model_end = time.time()
    print("\nModel trained. Elapse time (s): ", (model_end - model_start))

# REF: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
def save_model(model, model_name):
    fn_json = os.path.join(path_models, model_name + ".json")
    fn_weights = os.path.join(path_models, model_name + ".h5")

    # serialize model to JSON
    model_json = model.to_json()
    with open(fn_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(fn_weights)

    print("Saved model " + model_name + " to disk")

def load_model(model_name):
    fn_json = os.path.join(path_models, model_name + ".json")
    fn_weights = os.path.join(path_models, model_name + ".h5")

    json_file = open(fn_json, 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(fn_weights)
    loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Loaded model " + model_name + " from disk")
    return loaded_model

def evaluate_model(model):
    # Params
    verbose = 1
    workers = cpu_count()
    _, train_accuracy = model.evaluate_generator(train_it, verbose=verbose, workers=workers)
    _, val_accuracy = model.evaluate_generator(val_it, verbose=verbose, workers=workers)
    _, test1_accuracy = model.evaluate_generator(test1_it, verbose=verbose, workers=workers)
    _, test2_accuracy = model.evaluate_generator(test2_it, verbose=verbose, workers=workers)

    print("Train accuracy = " , train_accuracy)
    print("Val accuracy = ", val_accuracy)
    print("Test1 accuracy = ", test1_accuracy)
    print("Test2 accuracy = ", test2_accuracy)

# Main
model_name = "mini5b_test4b_1"
model = create_model()
train_model(model)
save_model(model, model_name)
#model = load_model(model_name)
evaluate_model(model)
