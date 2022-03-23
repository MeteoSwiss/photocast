# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Create hourly nowcast visualizations

from datetime import timedelta
import numpy as np
import os
from pyreadr import read_r
import tensorflow as tf

from data.image import write_png
from setup.tf_setup import tf_setup

from model import generator
from model.generator import noise_dim
from util.path import experiment_path

tf_setup(11000)
tf.random.set_seed(1)

run_path = experiment_path + '/run__/' + '20210716_080451'
checkpoint_nr = 167

# lead_times = [0, 10, 20, 30, 40, 50, 60]
lead_times = [0, 60, 120, 180, 240, 300, 360]
lead_times_descr = "hourly"

# %% Load data

index = read_r('data/nowcasting/cevio/index.RData')['index']
index_test = index[index.reference.dt.year >= 2020]

images = tf.convert_to_tensor(np.load('data/nowcasting/cevio/images_64_128.npy'))
weather = tf.convert_to_tensor(np.load('data/nowcasting/cevio/weather.npy'))

# %%  Load model

height = images.shape[1]
width = images.shape[2]
weather_features = weather.shape[1]

generator = generator.generator(height, width, weather_features)
checkpoint = tf.train.Checkpoint(generator=generator)
manager = tf.train.CheckpointManager(checkpoint, directory=run_path + '/checkpoint', max_to_keep=1)
# checkpoint.restore(manager.latest_checkpoint)
checkpoint.restore(run_path + '/checkpoint/ckpt-' + str(checkpoint_nr))


#%%

hours = [6, 10, 14]
stddev = [0, 0.1, 0.2, 0.5, 1.0]

for hour in hours:
    index_a = index_test[index_test.reference.dt.hour == hour]
    index_a = index_a[index_a.reference.dt.minute == 0]
    for a_ndx, a in index_a.iterrows():
        print(a.reference)

        target = []
        generated = [[] for s in range(len(stddev))]

        for minute in lead_times:
            b = index_test[index_test.identifier.eq(a.identifier) &
                           index_test.position.eq(a.position) &
                           index_test.reference.eq(a.reference + timedelta(minutes=minute))
                           ].squeeze()
            if len(b) == 0:
                break

            for sample in range(len(stddev)):
                noise = tf.random.normal([1, noise_dim], stddev=stddev[sample])
                generated[sample].append(generator([
                    noise,
                    tf.expand_dims(images[a.rowid, :, :, :], 0),
                    tf.expand_dims(tf.concat([weather[a.rowid, :], weather[b.rowid, :]], axis=0), 0)
                ])[0])

            target.append(images[b.rowid, :, :, :])

        comparison = tf.concat(target, axis=1)
        for sample in range(len(stddev)):
            comparison = tf.concat([comparison, tf.concat(generated[sample], axis=1)], axis=0)
        image_a_name = os.path.basename(a.path)[0:-5]
        write_png(comparison, run_path + '/nowcast-' + lead_times_descr
                  + '-chkpt=' + str(checkpoint.save_counter.numpy()) + '/' + image_a_name + '.png')
