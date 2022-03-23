# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import keras
from tensorflow import keras
from tensorflow.keras import layers

from ops.normalization import SpectralNormalization


def cbr(filters, name=None):

    block = keras.Sequential(name=name)
    block.add(layers.Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False,
                            kernel_constraint=SpectralNormalization()))
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())

    return block


def down(filters, name=None):

    block = keras.Sequential(name=name)
    block.add(layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding='same', use_bias=False,
                            kernel_constraint=SpectralNormalization()))
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())

    return block


def up(filters, name=None):

    block = keras.Sequential(name=name)
    block.add(layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding='same', use_bias=False,
                                     kernel_constraint=SpectralNormalization()))
    block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU())

    return block
