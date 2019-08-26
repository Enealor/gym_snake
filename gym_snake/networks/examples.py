import tensorflow as tf
import numpy as np
from baselines.a2c.utils import conv, fc, conv_to_fc

def snake_cnn(scaled_images, **conv_kwargs):
    """
    CNN based on Nature paper. Made smaller as there is less visual 
    information.
    """
    scaled_images = tf.cast(scaled_images, tf.float32)
    activ = tf.nn.relu
    h1 = activ(conv(scaled_images, 'c1', nf=16, rf=4, stride=1, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h1, 'c2', nf=64, rf=2, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
#    h3 = activ(conv(h2, 'c3', nf=128, rf=2, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    fl = conv_to_fc(h2)
    return activ(fc(fl, 'fc1', nh=128, init_scale=np.sqrt(2)))

def snake_cnn_big(scaled_images, **conv_kwargs):
    """
    A larger CNN.
    """
    scaled_images = tf.cast(scaled_images, tf.float32)
    activ = tf.nn.relu
    h1 = activ(conv(scaled_images, 'c1', nf=16, rf=4, stride=1, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h1, 'c2', nf=64, rf=2, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=128, rf=2, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    fl = conv_to_fc(h3)
    return activ(fc(fl, 'fc1', nh=256, init_scale=np.sqrt(2)))

def snake_nn(scaled_images, **conv_kwargs):
    """
    A neural network that takes the pixels in directly. Scales poorly with
    size.
    """
    scaled_images = tf.cast(scaled_images, tf.float32)
    activ = tf.nn.relu
    h1 = activ(fc(scaled_images, 'fc1', nh=64, init_scale=np.sqrt(2)))
    return activ(fc(h1, 'fc2', nh=128, init_scale=np.sqrt(2)))