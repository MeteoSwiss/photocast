# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from setup.tf_setup import tf_setup


def cutmix_maps(shape):

    batch_size, height, width, channels = shape
    maps = np.ones(shape)

    for bb in range(batch_size):
        r = np.sqrt(np.random.uniform())
        w = np.int(width * r)
        h = np.int(height * r)
        x = np.random.randint(width)
        y = np.random.randint(height)

        x1 = np.clip(x - w // 2, 0, width)
        y1 = np.clip(y - h // 2, 0, height)
        x2 = np.clip(x + w // 2, 0, width)
        y2 = np.clip(y + h // 2, 0, height)

        maps[bb, y1:y2, x1:x2, :] = 0
        if np.random.uniform() > 0.5:
            maps[bb, :, :, :] = 1 - maps[bb, :, :, :]

    return tf.convert_to_tensor(maps, dtype=tf.float32)


def cutmix_validate(height, width):

    maps = cutmix_maps([5, height, width, 1])
    for bb in range(maps.shape[0]):
        plt.imshow(maps[bb, :, :])
        plt.show()

    maps = cutmix_maps([10000, height, width, 1])
    areas = tf.math.reduce_mean(maps, [1, 2])
    plt.hist(areas.numpy())
    plt.show()


if __name__ == "__main__":
    tf_setup()
    cutmix_validate(128, 128)
