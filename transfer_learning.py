# import all required packages
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import keras
from keras.layers import Dense, Dropout, Flatten
import gc
from itertools import product
from multiprocessing import cpu_count
import winsound
import time
from keras.applications.nasnet import preprocess_input
from keras.applications.nasnet import NASNetMobile
from keras.models import Model

# indicate the starting number for the 1st run
session_num_init = 1

# indicate the number of Legos to classify
num_class = 3

# indicate the hyper-parameters interested to grid search during tunning
batch_sizes_ = [64]
res_xys = [224]

# set the max number of epochs each run will last
epoch_upper = 50

# initialize the number of max_queue_size and number of workers in order to better allocate CPU/GPU during training
max_queue_size_ = 2 * max(batch_sizes_)
workers_ = cpu_count()


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    """This function takes any base transferlearning model, fixes all existing weights,
    and add on new final dense layers with dropout rates and returns the full model ready to finetuning"""
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x)  # New FC layer, random init
        x = Dropout(dropout)(x)

    predictions = Dense(num_classes, activation='softmax')(x)  # New softmax layer

    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    return finetune_model


class TimeHistory(keras.callbacks.Callback):
    """This class generates and stores a list of seconds taken for each epoch for a run"""
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def train_test_model(hparams):
    """this function trains a Transfer Learning NASNetMobile based on the hparams chosen and stores accuracies,
    timetaken, parameters used for every dataset on tensorbaord and returns 4 metrics as a output"""
    # create a data generator
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # load and iterate datasets
    train_it = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini3_1R_R_R/train', shuffle=True,
                                           class_mode='categorical', batch_size=hparams[HP_BATCH_SIZE], target_size=(hparams[HP_RES], hparams[HP_RES]))

    test_it = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini3_1R_R_R/test1', shuffle=True,
                                          class_mode='categorical', batch_size=hparams[HP_BATCH_SIZE], target_size=(hparams[HP_RES], hparams[HP_RES]))

    test_it_2 = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini3_1R_R_R/test1', shuffle=True,
                                                    class_mode='categorical', batch_size=hparams[HP_BATCH_SIZE], target_size=(hparams[HP_RES], hparams[HP_RES]))

    val_it = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini3_1R_R_R/val', shuffle=True,
                                         class_mode='categorical', batch_size=hparams[HP_BATCH_SIZE], target_size=(hparams[HP_RES], hparams[HP_RES]))

    # load the base NASNetMobile model with no top layer and fixed imagenet weights
    base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(hparams[HP_RES], hparams[HP_RES], 3))

    # add final layers with specified dropouts at the end of the model
    model = build_finetune_model(base_model, dropout=hparams[HP_DROPOUT_DLS], fc_layers=[hparams[HP_NUM_UNITS_DLl], hparams[HP_NUM_UNITS_DL2]], num_classes=num_class)

    adam = keras.optimizers.Adam(learning_rate=hparams[HP_OPTIMIZER_PARAM], beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # print out model summary
    print(model.summary())

    # added a callbacks to store time taken to train each epoch
    time_callback = TimeHistory()

    # added a callbacks to store all settings and metrics on Tensorboard
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs\\hparam_tuning_DL_3\\" + str(run_name))

    # added a callbacks to save best models automatically
    filepath = "-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=('C:/Users/Grace/PycharmProjects/Jimi/logs/hparam_tuning_DL_3/' + 'model_' + str(run_name) + filepath),
                                                          monitor='val_accuracy', verbose=1, save_best_only=True)

    # added a callbacks to early-stop training if no future improvement
    es_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=max(int(epoch_upper/2), 5), verbose=1)

    # fit, train, and evaluate all 4 datasets
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
                                                 workers=workers_,
                                                 max_queue_size=max_queue_size_,
                                                 use_multiprocessing=False,
                                                 verbose=1
                                                 )
    _, val_accuracy = model.evaluate_generator(generator=validation_batch_generator,
                                               workers=workers_,
                                               max_queue_size=max_queue_size_,
                                               use_multiprocessing=False,
                                               verbose=1
                                               )
    _, test_accuracy = model.evaluate_generator(generator=test1_batch_generator,
                                                workers=workers_,
                                                max_queue_size=max_queue_size_,
                                                use_multiprocessing=False,
                                                verbose=1
                                                )
    _, test_real_accuracy = model.evaluate_generator(generator=test2_batch_generator,
                                                     workers=workers_,
                                                     max_queue_size=max_queue_size_,
                                                     use_multiprocessing=False,
                                                     verbose=1
                                                     )
    return train_accuracy, val_accuracy, test_accuracy, test_real_accuracy, time_callback.times

def run(run_dir, hparams):
    """This function runs train_test_model function above for each pair of hyper-parameter combinations(run),
     and stores final metrics on tensorboard"""
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
    """This function first setup the hyperparameters interested to tune, and calls the function run and
     train_test_model to train each combination and store all results to tensortboard"""

    # define all hyper-parameters interested to tune and all metrics interested to measure and store
    global HP_NUM_UNITS_DLl, HP_DROPOUT_DLS, HP_NUM_UNITS_DL2
    global HP_OPTIMIZER_PARAM, METRIC_TRAIN_ACCURACY, METRIC_VAL_ACCURACY, METRIC_TEST1_ACCURACY, METRIC_TEST2_ACCURACY, METRIC_TIME, HP_BATCH_SIZE, HP_RES
    global session_num, run_name

    HP_NUM_UNITS_DLl = hp.HParam('dl1_num_units', hp.Discrete([128]))
    HP_DROPOUT_DLS = hp.HParam('dl1&2_dropout', hp.Discrete([0.25]))
    HP_NUM_UNITS_DL2 = hp.HParam('dl2_num_units', hp.Discrete([64]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete(batch_sizes_))
    HP_RES = hp.HParam('res', hp.Discrete(res_xys))

    HP_OPTIMIZER_PARAM = hp.HParam('optimizer parameter', hp.Discrete([0.001]))
    METRIC_TRAIN_ACCURACY = 'Train Accuracy'
    METRIC_VAL_ACCURACY = 'Validation Accuracy'
    METRIC_TEST1_ACCURACY = 'Test1 Accuracy'
    METRIC_TEST2_ACCURACY = 'Test2 Accuracy'

    METRIC_TIME = 'average seconds per epoch'

    # initialize all tunable hyper-parameters and saveable metrics on Tensorboard
    with tf.summary.create_file_writer('logs/hparam_tuning_DL_3').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS_DLl, HP_DROPOUT_DLS, HP_NUM_UNITS_DL2, HP_BATCH_SIZE, HP_RES,
                     HP_OPTIMIZER_PARAM],
            metrics=[hp.Metric(METRIC_VAL_ACCURACY, display_name='Validation Accuracy'),
                     hp.Metric(METRIC_TIME, display_name='average seconds per epoch')]
        )

    session_num = session_num_init

    # grid search of all possible unique combinations of hyper-parameters
    for a,b,c,d,e,f in product(HP_NUM_UNITS_DLl.domain.values, HP_DROPOUT_DLS.domain.values, HP_NUM_UNITS_DL2.domain.values,
                               HP_OPTIMIZER_PARAM.domain.values, HP_BATCH_SIZE.domain.values, HP_RES.domain.values):
        hparams = {HP_NUM_UNITS_DLl: a, HP_DROPOUT_DLS: b, HP_NUM_UNITS_DL2: c,
                   HP_OPTIMIZER_PARAM: d, HP_BATCH_SIZE: e, HP_RES: f}
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning_DL_3/' + run_name, hparams)
        session_num += 1

        # clear all unneeded memories
        gc.collect()

# this will run main function followed by a sound alert when grid search is finished
if __name__ == '__main__':
    main()

    # sound alert when finished
    duration = 750  # milliseconds
    for freq in range(200, 800, 50):
        winsound.Beep(freq, duration)
        time.sleep(0.5)


# this is the command to run to see any Tensorboard results from a grid search:
# tensorboard --logdir=logs/hparam_tuning_DL_3