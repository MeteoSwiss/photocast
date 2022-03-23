# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Load and resize all images

import numpy as np
from pyreadr import read_r
import tensorflow as tf

from data.image import load_jpeg
from setup.tf_setup import tf_setup

height = 64
width = 128

tf_setup()

index = read_r('data/nowcasting/cevio/index.RData')['index']

images_list = []
for ii in range(index.shape[0]):
    print(ii)
    image = load_jpeg(index.iloc[ii].loc['path'])
    image = tf.image.resize(image, [height, width], tf.image.ResizeMethod.BICUBIC, antialias=True)
    images_list.append(image)

with tf.device('/cpu:0'):
    images = tf.stack(images_list)

np.save('data/nowcasting/cevio/images_' + str(height) + '_' + str(width) + '.npy', images.numpy())
