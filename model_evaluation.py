# import all required packages
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import cv2
from multiprocessing import cpu_count

# indicate the batch size and res for the model interested to evaluate against
batch_size = 64
res_xy = 64

# initialize the number of max_queue_size and number of workers in order to better allocate CPU/GPU during training
max_queue_size_ = 2 * batch_size
workers_ = cpu_count()


def crop_center(img, cropx, cropy):
    """This function crops any image by a factor of cropx and cropy and returns the cropped image unscaled"""
    y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    new_image = img[starty:starty+cropy, startx:startx+cropx]
    return cv2.resize(new_image, dsize=(y, x))


def convert_black_to_white_v2(batchX, crop_factor=0.9, noise_factor=0.2, crop=0, noise=0):
    """This function will convert all generated image's backgrounds from black to white and allow optional noise
    addition or additional image center crop on the generated images if specified"""
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
    """this function loads and evaluates any model and returns accuracies for each test set as outputs"""

    # create a data generator
    datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=convert_black_to_white_v2)

    # load and iterate datasets
    test_it = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini3_6MINR_R_R/val', shuffle=True,
                                          class_mode='categorical', batch_size=batch_size, target_size=(res_xy, res_xy),
                                          color_mode='grayscale')

    test_it_2 = datagen.flow_from_directory('C:/Users/Grace/PycharmProjects/Jimi/Dataset_mini3_6MINR_R_R/test1', shuffle=True,
                                                    class_mode='categorical', batch_size=batch_size, target_size=(res_xy, res_xy),
                                                    color_mode='grayscale')
    # load model
    model = load_model(file_)

    # evaluate all test sets
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


# this will load the model specified evaluate and print its results
if __name__ == '__main__':
    dir = "logs/hparam_tuning_DL_3_II/model_run-9-28-0.89.hdf5"
    print(main(dir))