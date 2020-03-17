from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import cv2
from multiprocessing import cpu_count
import glob
import os

"""
AAA
"""
batch_size = 64
res_xy = 64
max_queue_size_ = 2 * batch_size
workers_ = cpu_count()


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


def main(file_):
    """this function trains a CNN based on the hparams chosen and outputs accuracies for every dataset"""
    # create a data generator
    datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=convert_black_to_white_v2)

    test_it = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini3_6MINR_R_R/val', shuffle=True,
                                          class_mode='categorical', batch_size=batch_size, target_size=(res_xy, res_xy),
                                          color_mode='grayscale')

    test_it_2 = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini3_6MINR_R_R/test1', shuffle=True,
                                                    class_mode='categorical', batch_size=batch_size, target_size=(res_xy, res_xy),
                                                    color_mode='grayscale')

    model = load_model(file_)

    test1_batch_generator = test_it
    test2_batch_generator = test_it_2

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
    return test_accuracy, test_real_accuracy


if __name__ == '__main__':
    dir = "logs/hparam_tuning_DL_3_II/model_run-9-28-0.89.hdf5"
    print(main(dir))