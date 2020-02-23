# Import modules
import numpy as np
import matplotlib
#matplotlib.use('agg')
#%matplotlib inline
import matplotlib.pyplot as plt

import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from multiprocessing import cpu_count

# Record start time for the whole module
start = time.time()

# create generator
datagen = ImageDataGenerator(rescale=1. / 255)

# prepare an iterators for each dataset
dim_xy = 64  #a:28
target_size = (dim_xy,dim_xy)
batch_size = 64  #a:28
seed = 42
interpolation = 'bicubic'
color_mode='grayscale'
class_mode='categorical'

train_it = datagen.flow_from_directory('../Dataset_mini5/train/', class_mode=class_mode, target_size=target_size, color_mode=color_mode, batch_size=batch_size, seed=seed)
val_it = datagen.flow_from_directory('../Dataset_mini5/val/', class_mode=class_mode, target_size=target_size, color_mode=color_mode, batch_size=batch_size, seed=seed)
test1_it = datagen.flow_from_directory('../Dataset_mini5/test1/', class_mode=class_mode, target_size=target_size, color_mode=color_mode, batch_size=batch_size, seed=seed)

# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
#plt.imshow(batchX[2,:,:,:].reshape(28,28), cmap='binary')

def create_model():
    model = Sequential()

    #1.
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(dim_xy, dim_xy, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))

    #2.
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))
    model.add(Dropout(0.25))

    #3.
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))  #a:512
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu')) #a:1024
    model.add(BatchNormalization())
    model.add(Dropout(0.50))                 #a:0.50

    #4.
    model.add(Dense(5, activation='softmax'))

    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return (model)


model = create_model()

model_start = time.time()
history = model.fit_generator(
                            generator=train_it,
                            validation_data=val_it,
                            epochs=3,
                            max_queue_size=32,
                            workers=cpu_count(),
                            use_multiprocessing = True,
                            )
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=12, batch_size=128, callbacks=[tb_callback])
model_end = time.time()
print("\nModel trained. Elapse time (s): ", (model_end - model_start))

score = model.evaluate(X_test, y_test)
print("Model score (on test set):", score)
end = time.time()

