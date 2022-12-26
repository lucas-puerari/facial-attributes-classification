import os
import PIL
import json
import numpy as np
import tensorflow as tf

from utils.augmentation import random_rotation, random_gaussian_blur


def preparing(image_filepath, labels_filepath):
    filepath = image_filepath.decode('utf-8')
    filename = os.path.basename(filepath)

    image = PIL.Image.open(filepath)
    image = image.convert('RGB')
    image = np.asarray(image).astype('float32')

    with open(labels_filepath.decode('utf-8')) as f:
        labels_file = json.load(f)

    labels = list(labels_file[filename].values())
    labels = np.asarray(labels).astype('float32')

    return image, labels


def preprocessing(image, image_size, labels):
    # image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.resize_with_pad(image, image_size, image_size)
    return image, labels


def augmentation(image, labels):
    # Color
    image = tf.image.random_contrast(image, 0.5, 0.9)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    # Geometric
    image = tf.image.random_flip_left_right(image)
    image = random_rotation(image, 10)
    image = random_gaussian_blur(image)
    return image, labels


def normalization(image, labels):
    image = tf.math.divide(image, 255)
    return image, labels    
