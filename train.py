
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils import read_csv, check_img, load_image_train, load_image_test, plotImage, display

CSV_PATH = 'csv_and_images/list.csv'
IMAGE_CONTAINER = 'csv_and_images'
IMAGE_FORMAT ='jpg'

IMG_HEIGHT = 256
IMG_WIDTH = 256

CROP_HEIGHT = 256
CROP_WIDTH = 256


if __name__ == '__main__':

    df = read_csv(CSV_PATH)
    img_paths = df['Image']
    img_paths = check_img(img_paths, IMAGE_CONTAINER, IMAGE_FORMAT)

    mask_paths = df['Mask']
    mask_paths = check_img(mask_paths, IMAGE_CONTAINER, IMAGE_FORMAT)

    fold_index = df['FoldIndex']

    # print(fold_index.values)


    train_dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    train_dataset = train_dataset.shuffle(1024)
    train_dataset = train_dataset.map(load_image_train,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Show the training set
    for img, mask in train_dataset.take(5):
        print("Img: {}, Mask: {}".format(type(img), mask.shape))
        plotImage(img)
        # display([img, mask])

    train_dataset = train_dataset.batch(32)