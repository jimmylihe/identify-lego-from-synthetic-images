from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D, BatchNormalization, Activation
import gc
from itertools import product
import numpy as np
import cv2
from multiprocessing import cpu_count
import winsound
import time

"""
AAA
"""
session_num_init = 1
num_class = 3
batch_sizes_ = [16, 64, 256]
res_xys = [16, 32, 64]
epoch_upper = 50
max_queue_size_ = 2 * max(batch_sizes_)
workers_ = cpu_count()


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    new_image = img[starty:starty+cropy, startx:startx+cropx]
    return cv2.resize(new_image, dsize=(y, x))


def convert_black_to_white_v2(batchX, crop_factor=0.9, noise_factor=0.2, crop=0, noise=0):
    y, x, _ = batchX.shape
    batchX = batchX.reshape((y, x))
    if batchX[0, 0] == 0.0:
        batchX = np.where(batchX <= 30, 255, batchX)
        if noise == 1:
            batchX = batchX * (1 - noise_factor) + np.random.random((y, x)) * 255 * noise_factor
        if crop == 1:
            batchX = crop_center(batchX, int(y*crop_factor), int(x*crop_factor))
    batchX_new = batchX.reshape((y, x, 1))
    return batchX_new


def train_test_model(hparams):
    """this function trains a CNN based on the hparams chosen and outputs accuracies for every dataset"""
    # create a data generator
    datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=convert_black_to_white_v2)

    # load and iterate datasets
    train_it = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini3_6MINR_R_R/train', shuffle=True,
                                           class_mode='categorical', batch_size=hparams[HP_BATCH_SIZE], target_size=(hparams[HP_RES], hparams[HP_RES]),
                                           color_mode='grayscale')

    test_it = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini3_6MINR_R_R/test1', shuffle=True,
                                          class_mode='categorical', batch_size=hparams[HP_BATCH_SIZE], target_size=(hparams[HP_RES], hparams[HP_RES]),
                                          color_mode='grayscale')

    test_it_2 = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini3_6MINR_R_R/test1', shuffle=True,
                                                    class_mode='categorical', batch_size=hparams[HP_BATCH_SIZE], target_size=(hparams[HP_RES], hparams[HP_RES]),
                                                    color_mode='grayscale')

    val_it = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini3_6MINR_R_R/val', shuffle=True,
                                         class_mode='categorical', batch_size=hparams[HP_BATCH_SIZE], target_size=(hparams[HP_RES], hparams[HP_RES]),
                                         color_mode='grayscale')

    model = Sequential()

    model.add(Conv2D(hparams[HP_NUM_UNITS_CLl], kernel_size=hparams[HP_KERNEL_SIZE_CLl], strides=hparams[HP_STRIDES_CLl], input_shape=(hparams[HP_RES], hparams[HP_RES], 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(hparams[HP_NUM_UNITS_CL2], kernel_size=hparams[HP_KERNEL_SIZE_CL2], strides=hparams[HP_STRIDES_CL2], padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=hparams[HP_POOL_SIZE_PL1]))
    model.add(Dropout(hparams[HP_DROPOUT_CL2]))

    model.add(Conv2D(hparams[HP_NUM_UNITS_CL3], kernel_size=hparams[HP_KERNEL_SIZE_CL3], strides=hparams[HP_STRIDES_CL3], padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(hparams[HP_NUM_UNITS_CL4], kernel_size=hparams[HP_KERNEL_SIZE_CL4], strides=hparams[HP_STRIDES_CL4], padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=hparams[HP_POOL_SIZE_PL2]))
    model.add(Dropout(hparams[HP_DROPOUT_CL4]))

    model.add(Flatten())
    model.add(Dense(hparams[HP_NUM_UNITS_DLl]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(hparams[HP_DROPOUT_DL1]))

    model.add(Dense(hparams[HP_NUM_UNITS_DL2]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(hparams[HP_DROPOUT_DL2]))

    model.add(Dense(num_class, activation='softmax'))
    adam = keras.optimizers.Adam(learning_rate=hparams[HP_OPTIMIZER_PARAM], beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    time_callback = TimeHistory()
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs\\hparam_tuning_DL_3\\" + str(run_name))
    filepath = "-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=('C:/Users/Grace/PycharmProjects/Jimi/logs/hparam_tuning_DL_3/' + 'model_' + str(run_name) + filepath),
                                                          monitor='val_accuracy', verbose=1, save_best_only=True)
    es_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=max(int(epoch_upper/2), 5), verbose=1)


    train_batch_generator = train_it
    validation_batch_generator = val_it
    test1_batch_generator = test_it
    test2_batch_generator = test_it_2
    model.fit_generator(generator=train_batch_generator,
                        epochs=epoch_upper,
                        verbose=2,
                        use_multiprocessing=False,
                        workers=workers_,
                        max_queue_size=max_queue_size_,
                        validation_data=validation_batch_generator,
                        callbacks=[time_callback, tensorboard_callback, checkpoint_callback, es_callback]
                        )
    _, train_accuracy = model.evaluate_generator(generator=train_batch_generator,
                                                 max_queue_size=max_queue_size_,
                                                 workers=workers_,
                                                 use_multiprocessing=False,
                                                 verbose=1
                                                 )
    _, val_accuracy = model.evaluate_generator(generator=validation_batch_generator,
                                               max_queue_size=max_queue_size_,
                                               workers=workers_,
                                               use_multiprocessing=False,
                                               verbose=1
                                               )
    _, test_accuracy = model.evaluate_generator(generator=test1_batch_generator,
                                                max_queue_size=max_queue_size_,
                                                workers=workers_,
                                                use_multiprocessing=False,
                                                verbose=1
                                                )
    _, test_real_accuracy = model.evaluate_generator(generator=test2_batch_generator,
                                                     max_queue_size=max_queue_size_,
                                                     workers=workers_,
                                                     use_multiprocessing=False,
                                                     verbose=1
                                                     )
    return train_accuracy, val_accuracy, test_accuracy, test_real_accuracy, time_callback.times

def run(run_dir, hparams):
    """run model and store accuracies"""
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        train_accuracy, val_accuracy, test_accuracy, test_real_accuracy, times = train_test_model(hparams)
        average_time = sum(times) / len(times)
        print(train_accuracy, val_accuracy, test_accuracy, test_real_accuracy, average_time)
        tf.summary.scalar(METRIC_TRAIN_ACCURACY, train_accuracy, step=session_num)
        tf.summary.scalar(METRIC_VAL_ACCURACY, val_accuracy, step=session_num)
        tf.summary.scalar(METRIC_TEST1_ACCURACY, test_accuracy, step=session_num)
        tf.summary.scalar(METRIC_TEST2_ACCURACY, test_real_accuracy, step=session_num)
        tf.summary.scalar(METRIC_TIME, average_time, step=session_num)
        print("this run is now finished!")

def main():
    # initialize parameters to tune
    global HP_NUM_UNITS_CLl, HP_KERNEL_SIZE_CLl, HP_STRIDES_CLl, HP_NUM_UNITS_CL2, HP_KERNEL_SIZE_CL2, HP_STRIDES_CL2, HP_POOL_SIZE_PL1, HP_DROPOUT_CL2
    global HP_NUM_UNITS_CL3, HP_KERNEL_SIZE_CL3, HP_STRIDES_CL3, HP_NUM_UNITS_CL4, HP_KERNEL_SIZE_CL4, HP_STRIDES_CL4, HP_POOL_SIZE_PL2, HP_DROPOUT_CL4
    global HP_NUM_UNITS_DLl, HP_DROPOUT_DL1, HP_NUM_UNITS_DL2, HP_DROPOUT_DL2
    global HP_OPTIMIZER_PARAM, METRIC_TRAIN_ACCURACY, METRIC_VAL_ACCURACY, METRIC_TEST1_ACCURACY, METRIC_TEST2_ACCURACY, METRIC_TIME, HP_BATCH_SIZE, HP_RES
    global session_num, run_name

    HP_NUM_UNITS_CLl = hp.HParam('cl1_num_units', hp.Discrete([32]))
    HP_KERNEL_SIZE_CLl = hp.HParam('cl1_kernel_size', hp.Discrete([3]))
    HP_STRIDES_CLl = hp.HParam('cl1_strides', hp.Discrete([1]))
    HP_NUM_UNITS_CL2 = hp.HParam('cl2_num_units', hp.Discrete([32]))
    HP_KERNEL_SIZE_CL2 = hp.HParam('cl2_kernel_size', hp.Discrete([3]))
    HP_STRIDES_CL2 = hp.HParam('cl2_strides', hp.Discrete([1]))
    HP_POOL_SIZE_PL1 = hp.HParam('pl1_pool_size', hp.Discrete([2]))
    HP_DROPOUT_CL2 = hp.HParam('cl2_dropout', hp.Discrete([0.25]))

    HP_NUM_UNITS_CL3 = hp.HParam('cl3_num_units', hp.Discrete([64]))
    HP_KERNEL_SIZE_CL3 = hp.HParam('cl3_kernel_size', hp.Discrete([3]))
    HP_STRIDES_CL3 = hp.HParam('cl3_strides', hp.Discrete([1]))
    HP_NUM_UNITS_CL4 = hp.HParam('cl4_num_units', hp.Discrete([64]))
    HP_KERNEL_SIZE_CL4 = hp.HParam('cl4_kernel_size', hp.Discrete([3]))
    HP_STRIDES_CL4 = hp.HParam('cl4_strides', hp.Discrete([1]))
    HP_POOL_SIZE_PL2 = hp.HParam('pl2_pool_size', hp.Discrete([2]))
    HP_DROPOUT_CL4 = hp.HParam('cl4_dropout', hp.Discrete([0.25]))

    HP_NUM_UNITS_DLl = hp.HParam('dl1_num_units', hp.Discrete([128]))
    HP_DROPOUT_DL1 = hp.HParam('dl1_dropout', hp.Discrete([0.25]))
    HP_NUM_UNITS_DL2 = hp.HParam('dl2_num_units', hp.Discrete([64]))
    HP_DROPOUT_DL2 = hp.HParam('dl2_dropout', hp.Discrete([0.5]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete(batch_sizes_))
    HP_RES = hp.HParam('res', hp.Discrete(res_xys))

    HP_OPTIMIZER_PARAM = hp.HParam('optimizer parameter', hp.Discrete([0.001]))
    METRIC_TRAIN_ACCURACY = 'Train Accuracy'
    METRIC_VAL_ACCURACY = 'Validation Accuracy'
    METRIC_TEST1_ACCURACY = 'Test1 Accuracy'
    METRIC_TEST2_ACCURACY = 'Test2 Accuracy'

    METRIC_TIME = 'average seconds per epoch'

    with tf.summary.create_file_writer('logs/hparam_tuning_DL_3').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS_CLl, HP_KERNEL_SIZE_CLl, HP_STRIDES_CLl, HP_NUM_UNITS_CL2, HP_KERNEL_SIZE_CL2, HP_STRIDES_CL2, HP_POOL_SIZE_PL1, HP_DROPOUT_CL2,
                     HP_NUM_UNITS_CL3, HP_KERNEL_SIZE_CL3, HP_STRIDES_CL3, HP_NUM_UNITS_CL4, HP_KERNEL_SIZE_CL4, HP_STRIDES_CL4, HP_POOL_SIZE_PL2, HP_DROPOUT_CL4,
                     HP_NUM_UNITS_DLl, HP_DROPOUT_DL1, HP_NUM_UNITS_DL2, HP_DROPOUT_DL2, HP_BATCH_SIZE, HP_RES,
                     HP_OPTIMIZER_PARAM],
            metrics=[hp.Metric(METRIC_VAL_ACCURACY, display_name='Validation Accuracy'),
                     hp.Metric(METRIC_TIME, display_name='average seconds per epoch')]
        )

    session_num = session_num_init
    for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w in product(HP_NUM_UNITS_CLl.domain.values, HP_KERNEL_SIZE_CLl.domain.values, HP_STRIDES_CLl.domain.values, HP_NUM_UNITS_CL2.domain.values, HP_KERNEL_SIZE_CL2.domain.values, HP_STRIDES_CL2.domain.values, HP_POOL_SIZE_PL1.domain.values, HP_DROPOUT_CL2.domain.values,
                                                                 HP_NUM_UNITS_CL3.domain.values, HP_KERNEL_SIZE_CL3.domain.values, HP_STRIDES_CL3.domain.values, HP_NUM_UNITS_CL4.domain.values, HP_KERNEL_SIZE_CL4.domain.values, HP_STRIDES_CL4.domain.values, HP_POOL_SIZE_PL2.domain.values, HP_DROPOUT_CL4.domain.values,
                                                                 HP_NUM_UNITS_DLl.domain.values, HP_DROPOUT_DL1.domain.values, HP_NUM_UNITS_DL2.domain.values, HP_DROPOUT_DL2.domain.values,
                                                                 HP_OPTIMIZER_PARAM.domain.values, HP_BATCH_SIZE.domain.values, HP_RES.domain.values):
        hparams = {HP_NUM_UNITS_CLl: a, HP_KERNEL_SIZE_CLl: b, HP_STRIDES_CLl: c, HP_NUM_UNITS_CL2: d,
                   HP_KERNEL_SIZE_CL2: e, HP_STRIDES_CL2: f, HP_POOL_SIZE_PL1: g, HP_DROPOUT_CL2: h,
                   HP_NUM_UNITS_CL3: i, HP_KERNEL_SIZE_CL3: j, HP_STRIDES_CL3: k, HP_NUM_UNITS_CL4: l,
                   HP_KERNEL_SIZE_CL4: m, HP_STRIDES_CL4: n, HP_POOL_SIZE_PL2: o, HP_DROPOUT_CL4: p,
                   HP_NUM_UNITS_DLl: q, HP_DROPOUT_DL1: r, HP_NUM_UNITS_DL2: s, HP_DROPOUT_DL2: t,
                   HP_OPTIMIZER_PARAM: u, HP_BATCH_SIZE: v, HP_RES: w}
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning_DL_3/' + run_name, hparams)
        session_num += 1
        gc.collect()

if __name__ == '__main__':
    main()

    # sound alert when finished
    duration = 750  # milliseconds
    for freq in range(200, 800, 50):
        winsound.Beep(freq, duration)
        time.sleep(0.5)


"""
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
from matplotlib import pyplot as plt
import numpy as np
first_image = batchX[0]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((128, 128))
plt.imshow(pixels, cmap='gray')
plt.show()
print(pixels)
"""


# tensorboard --logdir=logs/hparam_tuning_DL_3