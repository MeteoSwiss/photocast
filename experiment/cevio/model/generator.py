# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ops.block import cbr, down, up
from ops.normalization import SpectralNormalization

filters = [64,   128, 256, 512, 1024, 1024,   512, 768, 640, 448, 288,   352]
noise_dim = 100
noise_channels = 128


def generator(height, width, weather_features):

    weather_input = keras.Input(shape=weather_features*2, name='weather-input')
    weather = layers.RepeatVector(height*width)(weather_input)
    weather = layers.Reshape((height, width, weather_features*2))(weather)

    image_input = keras.Input(shape=[height, width, 3], name='image-input')
    inputs = layers.Concatenate(name='inputs-concat')([image_input, weather])

    block = cbr(filters[0], 'pre-cbr-1')(inputs)

    u_skip_layers = [block]
    for ll in range(1, len(filters) // 2):

        block = down(filters[ll], 'down_%s-down' % ll)(block)

        # Collect U-Net skip connections
        u_skip_layers.append(block)

    noise_input = keras.Input(shape=noise_dim, name='noise-input')
    height = block.shape[1]
    width = block.shape[2]
    noise = layers.Dense(height * width * noise_channels,
                         # kernel_constraint=SpectralNormalization()
                         )(noise_input)
    noise = layers.Reshape((height, width, -1))(noise)

    block = layers.Concatenate(name='add-noise')([block, noise])
    u_skip_layers.pop()

    for ll in range(len(filters) // 2, len(filters) - 1):

        block = up(filters[ll], 'up_%s-up' % (len(filters) - ll - 1))(block)

        # Connect U-Net skip
        block = layers.Concatenate(name='up_%s-concatenate' % (len(filters) - ll - 1))([block, u_skip_layers.pop()])

    block = cbr(filters[-1], 'post-cbr-1')(block)

    rgb = layers.Conv2D(
        filters=3, kernel_size=1, padding='same', activation='tanh',
        kernel_constraint=SpectralNormalization(),
        name='rgb-conv')(block)

    return tf.keras.Model(inputs=[noise_input, image_input, weather_input], outputs=rgb)
