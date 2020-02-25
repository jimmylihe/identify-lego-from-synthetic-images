from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D, BatchNormalization
import gc
from multiprocessing import cpu_count
from itertools import product

def train_test_model(hparams):
    """this function trains a CNN based on the hparams chosen and outputs accuracies for every dataset"""
    model = Sequential()
    model.add(Conv2D(hparams[HP_NUM_UNITS_CLl], kernel_size=hparams[HP_KERNEL_SIZE_CLl], strides=hparams[HP_STRIDES_CLl], activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(hparams[HP_NUM_UNITS_CL2], kernel_size=hparams[HP_KERNEL_SIZE_CL2], strides=hparams[HP_STRIDES_CL2], activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=hparams[HP_POOL_SIZE_PL1]))
    model.add(Dropout(hparams[HP_DROPOUT_CL2]))

    model.add(Conv2D(hparams[HP_NUM_UNITS_CL3], kernel_size=hparams[HP_KERNEL_SIZE_CL3], strides=hparams[HP_STRIDES_CL3], activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(hparams[HP_NUM_UNITS_CL4], kernel_size=hparams[HP_KERNEL_SIZE_CL4], strides=hparams[HP_STRIDES_CL4], activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=hparams[HP_POOL_SIZE_PL2]))
    model.add(Dropout(hparams[HP_DROPOUT_CL4]))

    model.add(Flatten())
    model.add(Dense(hparams[HP_NUM_UNITS_DLl], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hparams[HP_DROPOUT_DL1]))
    model.add(Dense(hparams[HP_NUM_UNITS_DL2], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hparams[HP_DROPOUT_DL2]))

    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer=hparams[HP_OPTIMIZER], loss='categorical_crossentropy', metrics=['accuracy'])

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs\\hparam_tuning_DL_5\\" + str(run_name), histogram_freq=1)
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=('C:/Users/Grace/PycharmProjects/Jimi/logs/hparam_tuning_DL_5/' + 'model_' + str(run_name) + '.h5'),
                                                             monitor='val_accuracy',
                                                             save_best_only=True)

    train_batch_generator = train_it
    validation_batch_generator = val_it
    test1_batch_generator = test_it
    test2_batch_generator = test_it_real
    model.fit_generator(generator=train_batch_generator,
                        epochs=5,
                        verbose=2,
                        use_multiprocessing=False,
                        workers=cpu_count(),
                        max_queue_size=32,
                        validation_data=validation_batch_generator,
                        callbacks=[tensorboard_callback, es_callback, checkpoint_callback])
    _, train_accuracy = model.evaluate_generator(generator=train_batch_generator,
                                           max_queue_size=32,
                                           workers=cpu_count(),
                                           use_multiprocessing=False,
                                           verbose=1
                                           )
    _, val_accuracy = model.evaluate_generator(generator=validation_batch_generator,
                                           max_queue_size=32,
                                           workers=cpu_count(),
                                           use_multiprocessing=False,
                                           verbose=1
                                           )
    _, test_accuracy = model.evaluate_generator(generator=test1_batch_generator,
                                           max_queue_size=32,
                                           workers=cpu_count(),
                                           use_multiprocessing=False,
                                           verbose=1
                                           )
    _, test_real_accuracy = model.evaluate_generator(generator=test2_batch_generator,
                                           max_queue_size=32,
                                           workers=cpu_count(),
                                           use_multiprocessing=False,
                                           verbose=1
                                           )
    print(model.summary())
    return train_accuracy, val_accuracy, test_accuracy, test_real_accuracy

def run(run_dir, hparams):
    """run model and store accuracies"""
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        train_accuracy, val_accuracy, test_accuracy, test_real_accuracy = train_test_model(hparams)
        print(train_accuracy, val_accuracy, test_accuracy, test_real_accuracy)
        tf.summary.scalar(METRIC_TRAIN_ACCURACY, train_accuracy, step=session_num)
        tf.summary.scalar(METRIC_VAL_ACCURACY, val_accuracy, step=session_num)
        tf.summary.scalar(METRIC_TEST1_ACCURACY, test_accuracy, step=session_num)
        tf.summary.scalar(METRIC_TEST2_ACCURACY, test_real_accuracy, step=session_num)

def main():
    global train_it, val_it, test_it, test_it_real, run_name, session_num
    # create a data generator
    datagen = ImageDataGenerator(rescale=1. / 255)

    # load and iterate training dataset
    train_it = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini5/train',
                                           class_mode='categorical', batch_size=64, seed=42, target_size=(28, 28),
                                           color_mode='grayscale')
    # load and iterate validation dataset
    val_it = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini5/val',
                                         class_mode='categorical', batch_size=64, seed=42, target_size=(28, 28),
                                         color_mode='grayscale')
    # load and iterate test dataset
    test_it = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini5/test1',
                                          class_mode='categorical', batch_size=64, seed=42, target_size=(28, 28),
                                          color_mode='grayscale')
    # load and iterate test dataset
    test_it_real = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini5/test2',
                                          class_mode='categorical', batch_size=64, seed=42, target_size=(28, 28),
                                          color_mode='grayscale')

    # initialize parameters to tune
    global HP_NUM_UNITS_CLl, HP_KERNEL_SIZE_CLl, HP_STRIDES_CLl, HP_NUM_UNITS_CL2, HP_KERNEL_SIZE_CL2, HP_STRIDES_CL2, HP_POOL_SIZE_PL1, HP_DROPOUT_CL2
    global HP_NUM_UNITS_CL3, HP_KERNEL_SIZE_CL3, HP_STRIDES_CL3, HP_NUM_UNITS_CL4, HP_KERNEL_SIZE_CL4, HP_STRIDES_CL4, HP_POOL_SIZE_PL2, HP_DROPOUT_CL4
    global HP_NUM_UNITS_DLl, HP_DROPOUT_DL1, HP_NUM_UNITS_DL2, HP_DROPOUT_DL2
    global HP_OPTIMIZER, METRIC_TRAIN_ACCURACY, METRIC_VAL_ACCURACY, METRIC_TEST1_ACCURACY, METRIC_TEST2_ACCURACY
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

    HP_NUM_UNITS_DLl = hp.HParam('dl1_num_units', hp.Discrete([512]))
    HP_DROPOUT_DL1 = hp.HParam('dl1_dropout', hp.Discrete([0.25]))
    HP_NUM_UNITS_DL2 = hp.HParam('dl2_num_units', hp.Discrete([1024]))
    HP_DROPOUT_DL2 = hp.HParam('dl2_dropout', hp.Discrete([0.5]))

    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['rmsprop']))
    METRIC_TRAIN_ACCURACY = 'Train Accuracy'
    METRIC_VAL_ACCURACY = 'Validation Accuracy'
    METRIC_TEST1_ACCURACY = 'Test1 Accuracy'
    METRIC_TEST2_ACCURACY = 'Test2 Accuracy'

    with tf.summary.create_file_writer('logs/hparam_tuning_DL_5').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS_CLl, HP_KERNEL_SIZE_CLl, HP_STRIDES_CLl, HP_NUM_UNITS_CL2, HP_KERNEL_SIZE_CL2, HP_STRIDES_CL2, HP_POOL_SIZE_PL1, HP_DROPOUT_CL2,
                     HP_NUM_UNITS_CL3, HP_KERNEL_SIZE_CL3, HP_STRIDES_CL3, HP_NUM_UNITS_CL4, HP_KERNEL_SIZE_CL4, HP_STRIDES_CL4, HP_POOL_SIZE_PL2, HP_DROPOUT_CL4,
                     HP_NUM_UNITS_DLl, HP_DROPOUT_DL1, HP_NUM_UNITS_DL2, HP_DROPOUT_DL2,
                     HP_OPTIMIZER],
            metrics=[hp.Metric(METRIC_VAL_ACCURACY, display_name='Validation Accuracy')],
        )

    session_num = 1
    for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u in product(HP_NUM_UNITS_CLl.domain.values, HP_KERNEL_SIZE_CLl.domain.values, HP_STRIDES_CLl.domain.values, HP_NUM_UNITS_CL2.domain.values, HP_KERNEL_SIZE_CL2.domain.values, HP_STRIDES_CL2.domain.values, HP_POOL_SIZE_PL1.domain.values, HP_DROPOUT_CL2.domain.values,
                                                           HP_NUM_UNITS_CL3.domain.values, HP_KERNEL_SIZE_CL3.domain.values, HP_STRIDES_CL3.domain.values, HP_NUM_UNITS_CL4.domain.values, HP_KERNEL_SIZE_CL4.domain.values, HP_STRIDES_CL4.domain.values, HP_POOL_SIZE_PL2.domain.values, HP_DROPOUT_CL4.domain.values,
                                                           HP_NUM_UNITS_DLl.domain.values, HP_DROPOUT_DL1.domain.values, HP_NUM_UNITS_DL2.domain.values, HP_DROPOUT_DL2.domain.values,
                                                           HP_OPTIMIZER.domain.values):
        hparams = {HP_NUM_UNITS_CLl: a, HP_KERNEL_SIZE_CLl: b, HP_STRIDES_CLl: c, HP_NUM_UNITS_CL2: d,
                   HP_KERNEL_SIZE_CL2: e, HP_STRIDES_CL2: f, HP_POOL_SIZE_PL1: g, HP_DROPOUT_CL2: h,
                   HP_NUM_UNITS_CL3: i, HP_KERNEL_SIZE_CL3: j, HP_STRIDES_CL3: k, HP_NUM_UNITS_CL4: l,
                   HP_KERNEL_SIZE_CL4: m, HP_STRIDES_CL4: n, HP_POOL_SIZE_PL2: o, HP_DROPOUT_CL4: p,
                   HP_NUM_UNITS_DLl: q, HP_DROPOUT_DL1: r, HP_NUM_UNITS_DL2: s, HP_DROPOUT_DL2: t,
                   HP_OPTIMIZER: u}
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning_DL_5/' + run_name, hparams)
        session_num += 1
        gc.collect()

if __name__ == '__main__':
    main()


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


# tensorboard --logdir=logs/hparam_tuning_DL_5