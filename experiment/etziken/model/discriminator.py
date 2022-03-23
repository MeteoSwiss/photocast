# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from tensorflow import keras
from tensorflow.keras import layers

from ops.block import cbr, down, up
from ops.normalization import SpectralNormalization

filters = [64,   128, 256, 512, 1024, 1024,   512, 768, 640, 448, 288,   352]
interpolation = 'nearest'


def discriminator(height, width, weather_features):

    weather_input = keras.Input(shape=weather_features * 2, name='weather-input')
    weather = layers.RepeatVector(height*width)(weather_input)
    weather = layers.Reshape((height, width, weather_features * 2))(weather)

    image_a_input = keras.Input(shape=[height, width, 3], name='image_a-input')
    image_b_input = keras.Input(shape=[height, width, 3], name='image_b-input')

    inputs = layers.Concatenate(name='inputs-concat')([image_a_input, weather, image_b_input])

    block = cbr(filters[0], 'pre-cbr-1')(inputs)

    u_skip_layers = [block]
    for ll in range(1, len(filters) // 2):

        block = down(filters[ll], 'down_%s-down' % ll)(block)

        # Collect U-Net skip connections
        u_skip_layers.append(block)

    label_global = layers.Conv2D(
        filters=1, kernel_size=1, padding='same',
        kernel_constraint=SpectralNormalization(),
        name='label_global')(block)
    u_skip_layers.pop()

    for ll in range(len(filters) // 2, len(filters) - 1):

        block = up(filters[ll], 'up_%s-up' % (len(filters) - ll - 1))(block)

        # Connect U-Net skip
        block = layers.Concatenate(name='up_%s-concatenate' % (len(filters) - ll - 1))([block, u_skip_layers.pop()])

    block = cbr(filters[-1], 'post-cbr-1')(block)

    label_pixel = layers.Conv2D(
        filters=1, kernel_size=1, padding='same',
        kernel_constraint=SpectralNormalization(),
        name='label_pixel')(block)

    return keras.Model(inputs=[image_a_input, weather_input, image_b_input], outputs=[label_global, label_pixel])
