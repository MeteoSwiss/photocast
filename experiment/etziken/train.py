# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

# Train the generator and discriminator networks

from contextlib import redirect_stdout
import datetime
import numpy as np
from pathlib import Path
import tensorflow as tf

from setup.tf_setup import tf_setup

from model import generator, discriminator
from util.path import experiment_path
from util.training import train_gan

tf_setup(26000)
tf.random.set_seed(1)

run_path = experiment_path + '/run__/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

Path(run_path).mkdir(parents=True, exist_ok=True)


# %% Profiling and Debugging

# tf.profiler.experimental.server.start(6009)
# tf.config.experimental_run_functions_eagerly(True)

# %% Create Datasets

batch_size = 50

images = tf.convert_to_tensor(np.load('data/nowcasting/etziken/images_64_128.npy'))
weather = tf.convert_to_tensor(np.load('data/nowcasting/etziken/weather.npy'))
pairs = np.load('data/nowcasting/etziken/pairs.npy').astype(int)

# images tensor with about 30000 samples leads to a libprotobuf error when creating the dataset iterator
images0 = images[0:20000, :, :, :]
images2 = images[20000:40000, :, :, :]
images4 = images[40000:, :, :, :]


def access_image(idx):
    if idx < 20000:
        return images0[idx, :, :, :]
    elif idx >= 20000 and idx < 40000:  # 20000 <= idx < 40000 leads to error
        return images2[idx - 20000, :, :, :]
    else:
        return images4[idx - 40000, :, :, :]


def pair(p):
    return access_image(p[0]), weather[p[0], :], access_image(p[1]), weather[p[1], :]


dataset_train = tf.data.Dataset.from_tensor_slices(pairs).map(pair, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
dataset_train = dataset_train.shuffle(100*batch_size, reshuffle_each_iteration=True).batch(batch_size) \
    .prefetch(tf.data.AUTOTUNE)

# %% Model

height = images.shape[1]
width = images.shape[2]
weather_features = weather.shape[1]

generator = generator.generator(height, width, weather_features)
optimizer_gen = tf.keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.0, beta_2=0.9)  # betas need to be floats, or checkpoint restoration fails

discriminator = discriminator.discriminator(height, width, weather_features)
optimizer_disc = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)

with open(run_path + '/generator_summary.txt', 'w') as handle:
    with redirect_stdout(handle):
        generator.summary()
tf.keras.utils.plot_model(generator, to_file=run_path + '/generator.png', show_shapes=True, dpi=96)

with open(run_path + '/discriminator_summary.txt', 'w') as handle:
    with redirect_stdout(handle):
        discriminator.summary()
tf.keras.utils.plot_model(discriminator, to_file=run_path + '/discriminator.png', show_shapes=True, dpi=96)


# %% Train

train_gan(generator, optimizer_gen, discriminator, optimizer_disc, dataset_train, run_path)
