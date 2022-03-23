# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Create pairs of real and generated images

from datetime import timedelta
import numpy as np
import os
from pyreadr import read_r
import random
import tensorflow as tf

from data.image import write_png
from setup.tf_setup import tf_setup

from model import generator
from model.generator import noise_dim
from util.path import experiment_path

tf_setup(11000)

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

run_path = experiment_path + '/run__/' + '20210826_071734'
checkpoint_nr = 312

# %% Load data

index = read_r('data/nowcasting/etziken/index.RData')['index']
index = index[index.reference.dt.year >= 2020]  # only consider test data
index = index[index.reference.dt.hour >= 6]
index = index[index.reference.dt.hour <= 14]

images = tf.convert_to_tensor(np.load('data/nowcasting/etziken/images_64_128.npy'))
weather = tf.convert_to_tensor(np.load('data/nowcasting/etziken/weather.npy'))

# %%  Load model

height = images.shape[1]
width = images.shape[2]
weather_features = weather.shape[1]

generator = generator.generator(height, width, weather_features)
checkpoint = tf.train.Checkpoint(generator=generator)
manager = tf.train.CheckpointManager(checkpoint, directory=run_path + '/checkpoint', max_to_keep=1)
checkpoint.restore(run_path + '/checkpoint/ckpt-' + str(checkpoint_nr))


#%%

nsamples = 75
randprefix = np.random.permutation(range(0, nsamples*2))

ii = 0
while ii < nsamples:
    a = index.sample().squeeze()
    minutes = random.sample(range(0, 360, 10), 1)[0]

    b = index[index.identifier.eq(a.identifier) &
              index.position.eq(a.position) &
              index.reference.eq(a.reference + timedelta(minutes=minutes))
              ].squeeze()
    if len(b) == 0:
        continue

    noise = tf.random.normal([1, noise_dim], stddev=0)
    generated = generator([
        noise,
        tf.expand_dims(images[a.rowid, :, :, :], 0),
        tf.expand_dims(tf.concat([weather[a.rowid, :], weather[b.rowid, :]], axis=0), 0)
    ])[0]

    image_b_name = os.path.basename(b.path)[0:-5]
    write_png(generated, run_path + '/realism-chkpt=' + str(checkpoint.save_counter.numpy()) + '/' + str(randprefix[2*ii]) + '-' + image_b_name + '-generated.png')
    write_png(images[b.rowid, :, :, :], run_path + '/realism-chkpt=' + str(checkpoint.save_counter.numpy()) + '/' + str(randprefix[2*ii + 1]) + '-' + image_b_name + '-real.png')

    ii = ii + 1
