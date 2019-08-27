
import pandas as pd
import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

CSV_PATH = 'csv_and_images/list.csv'
IMAGE_CONTAINER = 'csv_and_images'
IMAGE_FORMAT ='jpg'

IMG_HEIGHT = 256
IMG_WIDTH = 256

CROP_HEIGHT = 256
CROP_WIDTH = 256

def load(image_A_file, image_B_file):
    image_A = tf.io.read_file(image_A_file)
    image_A = tf.image.decode_jpeg(image_A)
    image_A = tf.cast(image_A, tf.float32)

    image_B = tf.io.read_file(image_B_file)
    image_B = tf.image.decode_jpeg(image_B)
    image_B = tf.cast(image_B, tf.float32)

    return image_A, image_B

def resize(image_A, image_B, height, width):
    image_A = tf.image.resize(image_A, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_B = tf.image.resize(image_B, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image_A, image_B

def random_crop(image_A, image_B, height, width):
    stacked_image = tf.stack([image_A, image_B], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, height, width, 3])

    return cropped_image[0], cropped_image[1]


def normalize(image_A, image_B):
    image_A = (image_A / 127.5) - 1
    image_B = (image_B / 127.5) - 1

    return image_A, image_B

@tf.function
def random_jitter(image_A, image_B):
    # resizing to IMG_HEIGHT x IMG_WIDTH x 3
    image_A, image_B = resize(image_A, image_B, IMG_HEIGHT, IMG_WIDTH)

    # randomly cropping to CROP_HEIGHT x CROP_WIDTH x 3
    image_A, image_B = random_crop(image_A, image_B, CROP_HEIGHT, CROP_WIDTH)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        image_A = tf.image.flip_left_right(image_A)
        image_B = tf.image.flip_left_right(image_B)

    return image_A, image_B


def load_image_train(image_A_file, image_B_file):
    image, mask = load(image_A_file, image_B_file)
    image, mask = random_jitter(image, mask)
    image, mask = normalize(image, mask)

    return image, mask


def load_image_test(image_A_file, image_B_file):
    image, mask = load(image_A_file, image_B_file)
    image, mask = resize(image, mask, IMG_HEIGHT, IMG_WIDTH)
    image, mask = normalize(image, mask)

    return image, mask


def plotImage(img):
    img = tf.cast(((img + 1) * 127.5), tf.uint8)
    cv2.imshow("test", np.array(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def read_csv(file_path):
    try:

        df = pd.read_csv(file_path)

        return df

    except FileNotFoundError as e:
        print(e, "\nPlease check the csv filepath!")


def get_imgs_and_masks(image_paths, mask_paths):

    images = []
    masks = []
    for image_path in image_paths:
        # images.append(np.array(Image.open(image_path)))
        images.append(cv2.imread(image_path))

    for mask_path in mask_paths:
        # masks.append(np.array(Image.open(mask_path)))
        masks.append(cv2.imread(mask_path))

    return np.array(images), np.array(masks)

def check_img(image_paths, image_container, image_format):
    '''
    Check whether images exist in folder or not
    Note: all images should be in image container
    '''

    # check whether image format is already embedded in image paths (check ".")
    if image_format is None and "." not in image_paths[0]:

        raise FileNotFoundError("There no image format specified in image path {}. "
                                "Please specifiy image format".format(image_paths[0]))
    elif image_format is not None and "." not in image_paths[0]:

        print("Concatenate image format into image paths ...")

        image_paths = np.array([os.path.join(image_container,
                                                    image_path + "." + image_format)
                                       for image_path in image_paths])
    else:
        image_paths = np.array([os.path.join(image_container,
                                                    image_path) for image_path in image_paths])
        # print(image_paths)

    for path in image_paths:

        if not os.path.isfile(path):
            raise FileNotFoundError("Image file - {} not found! Please check!".format(path))

    print("All images exist in folder - {}".format(image_container))
    return image_paths


if __name__ == '__main__':
    df = read_csv(CSV_PATH)
    # print("Head:\n{}".format(df.head()))
    # print("Dtypes:\n{}".format(df.dtypes))
    # df['Image'] = pd.Categorical(df['Image'])
    # df['Image'] = df.Image.cat.codes
    # print("Image:\n{}".format(df['Image']))

    # img = df.pop('Image')
    img_paths = df['Image']
    img_paths = check_img(img_paths, IMAGE_CONTAINER, IMAGE_FORMAT)

    mask_paths = df['Mask']
    mask_paths = check_img(mask_paths, IMAGE_CONTAINER, IMAGE_FORMAT)

    # img, mask = load(img_paths[0], mask_paths[0])

    imgs, masks = get_imgs_and_masks(img_paths, mask_paths)
    # print("Images: {}".format(imgs[0].shape))
    # print("Masks: {}".format(masks.shape))

    train_dataset = tf.data.Dataset.from_tensor_slices((imgs, masks))
    for img, mask in train_dataset.take(5):
        print("Img: {}, Mask: {}".format(img.shape, mask.shape))