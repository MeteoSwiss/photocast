# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import tensorflow as tf


def load_jpeg(path):
    image = tf.io.read_file(path)
    jpeg = tf.image.decode_jpeg(image)
    jpeg = tf.cast(jpeg, tf.float32)
    jpeg = jpeg[:-10, :-10]/127.5 - 1  # cut away black bars for old cameras
    return jpeg


def write_png(image, path):
    image = (image + 1)*127.5
    image = tf.cast(image, tf.uint8)
    png = tf.image.encode_png(image)
    tf.io.write_file(path, png)
