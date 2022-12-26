import math
import tensorflow as tf
import tensorflow_addons as tfa


def random_rotation(image, angle, prob=0.5):
    random_prob = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=1,
        dtype=tf.float32
    )

    if random_prob < prob:
        lower = -angle * (math.pi / 180)
        upper = angle * (math.pi / 180)
        rotation = tf.random.uniform(
            shape=[],
            minval=lower,
            maxval=upper,
            dtype=tf.float32
        )
        image = tfa.image.rotate(image, rotation)

    return image


def random_gaussian_blur(image, prob=0.5):
    random_prob = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=1,
        dtype=tf.float32
    )

    if random_prob < prob:
        image = tfa.image.gaussian_filter2d(image)

    return image
