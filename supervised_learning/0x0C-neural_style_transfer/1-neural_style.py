#!/usr/bin/env python3
"""module"""


import tensorflow as tf
import numpy as np


class NST:
    """class"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """constructor"""
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or \
           style_image.shape[-1] != 3:
            raise TypeError('style_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 or \
           content_image.shape[-1] != 3:
            raise TypeError('content_image must be a numpy.ndarray with shape (h, w, 3)')
        if alpha < 0 or not type(alpha) in [float, int]:
            raise TypeError('alpha must be a non-negative number')
        if beta < 0 or not type(beta) in [float, int]:
            raise TypeError('beta must be a non-negative number')
        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        """method"""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or \
           image.shape[-1] != 3:
            raise TypeError('image must be a numpy.ndarray with shape (h, w, 3)')
        h, w, _ = image.shape
        if h == max(h, w):
            new_size = (512, int(512*w/h))
        else:
            new_size = (int(512*h/w), 512)
        reshape = tf.expand_dims(image, 0)
        resized = tf.image.resize_bicubic(reshape, new_size)
        scaled = tf.divide(resized, 255)
        scaled = tf.clip_by_value(scaled, 0., 1.)
        return scaled

    def load_model(self):
        """method"""
        base = tf.keras.applications.vgg19.VGG19(include_top=False)
        base.trainable = False
        base.save('base_model')
        avge = tf.keras.layers.AveragePooling2D
        avge_pool = {'MaxPooling2D': avge}
        base_model = tf.keras.models.load_model('base_model', custom_objects=avge_pool)
        self.style_layers.append(self.content_layer)
        out = [base_model.get_layer(layer).output for layer in self.style_layers]
        return tf.keras.Model(inputs=base_model.inputs, outputs=out)
