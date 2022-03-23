# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import tensorflow as tf


def tf_setup(memory_limit=8000):

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.set_logical_device_configuration(
            gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])
