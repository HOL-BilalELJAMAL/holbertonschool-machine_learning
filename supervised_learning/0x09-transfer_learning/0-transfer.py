#!/usr/bin/env python3
"""
0-transfer.py
Module that use transfer learning with DenseNet-169 to predict CIFAR-10 dataset
"""

import tensorflow as tf
import tensorflow.keras as k


def preprocess_data(X, Y):
    """
    Function that pre-processes the data for the model

    Args:
        X (numpy.ndarray) of shape (m, 32, 32, 3) containing
        the CIFAR 10 data, where m is the number of data points
        Y (numpy.ndarray) of shape (m,) containing the CIFAR 10 labels for X
    Returns:
        x_p (numpy.ndarray) containing the preprocessed X
        y_p (numpy.ndarray) containing the preprocessed Y
    """
    x_p = k.applications.densenet.preprocess_input(X)
    y_p = k.utils.to_categorical(Y, 10)
    return x_p, y_p


if __name__ == '__main__':
    (X_train, Y_train), (X_val, Y_val) = k.datasets.cifar10.load_data()

    X_train, Y_train = preprocess_data(X_train, Y_train)

    X_val, Y_val = preprocess_data(X_val, Y_val)

    initializer = k.initializers.he_normal(seed=None)

    input_shape_densenet = (224, 224, 3)

    densenet_model = k.applications.DenseNet169(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=input_shape_densenet,
        pooling=None
    )

    densenet_model.trainable = True

    for layer in densenet_model.layers:
        if 'conv5' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    input = k.Input(shape=(32, 32, 3))

    preprocess = k.layers.Lambda(
        lambda x: tf.image.resize_images(x, (224, 224)), name='lamb')(input)

    layer = densenet_model(inputs=preprocess)

    layer = k.layers.Flatten()(layer)

    layer = k.layers.BatchNormalization()(layer)

    layer = k.layers.Dense(units=256,
                           activation='relu',
                           kernel_initializer=initializer
                           )(layer)

    layer = k.layers.Dropout(0.4)(layer)

    layer = k.layers.BatchNormalization()(layer)

    layer = k.layers.Dense(units=128,
                           activation='relu',
                           kernel_initializer=initializer
                           )(layer)

    layer = k.layers.Dropout(0.4)(layer)

    layer = k.layers.Dense(units=10,
                           activation='softmax',
                           kernel_initializer=initializer
                           )(layer)

    model = k.models.Model(inputs=input, outputs=layer)

    model.compile(loss='binary_crossentropy',
                  optimizer=k.optimizers.Adam(),
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train, Y_train, epochs=20, validation_data=(
        X_val, Y_val), batch_size=32, verbose=1)

    model.save('cifar10.h5')
