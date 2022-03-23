# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import tensorflow as tf
import time

from data import image
from ops.cut_mix import cutmix_maps

from model.generator import noise_dim

bxe_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def gan_step(generator, optimizer_gen, discriminator, optimizer_disc,
             images_a, weathers_a, images_b, weathers_b, maps,
             summary_writer, step):

    noise = tf.random.normal([images_a.shape[0], noise_dim])
    noise2 = tf.random.normal([images_a.shape[0], noise_dim])
    weathers = tf.concat([weathers_a, weathers_b], axis=1)
    with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:
        generated = generator([noise, images_a, weathers])
        mixed = maps * images_b + (1 - maps) * generated

        discriminated_real_global, discriminated_real_pixel = discriminator([images_a, weathers, images_b])
        discriminated_fake_global, discriminated_fake_pixel = discriminator([images_a, weathers, generated])
        _, discriminated_mixed_pixel = discriminator([images_a, weathers, mixed])

        gen_fake_loss = bxe_loss(tf.ones_like(discriminated_fake_global), discriminated_fake_global) + \
                        bxe_loss(tf.ones_like(discriminated_fake_pixel), discriminated_fake_pixel)

        generated2 = generator([noise2, images_a, weathers])
        gen_similarity_loss = -tf.math.reduce_mean(tf.math.abs(generated - generated2))

        gen_loss = gen_fake_loss + gen_similarity_loss

        disc_real_loss = bxe_loss(tf.ones_like(discriminated_real_global), discriminated_real_global)
        disc_fake_loss = bxe_loss(tf.zeros_like(discriminated_fake_global), discriminated_fake_global)
        disc_mixed_loss = bxe_loss(maps[:, :, :, 0:1], discriminated_mixed_pixel)
        disc_loss = disc_real_loss + disc_fake_loss + disc_mixed_loss

    gradients_gen = tape_gen.gradient(gen_loss, generator.trainable_variables)
    optimizer_gen.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    gradients_disc = tape_disc.gradient(disc_loss, discriminator.trainable_variables)
    optimizer_disc.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_fake_loss', gen_fake_loss, step=step)
        tf.summary.scalar('gen_similarity_loss', gen_similarity_loss, step=step)
        tf.summary.scalar('disc_loss', disc_loss, step=step)


def train_gan(generator, optimizer_gen, discriminator, optimizer_disc, dataset_train, run_path):

    summary_writer = tf.summary.create_file_writer(run_path)
    epoch = tf.Variable(1, dtype='int64')
    step = tf.Variable(1, dtype='int64')

    ckpt = tf.train.Checkpoint(epoch=epoch, step=step,
                               generator=generator, optimizer_gen=optimizer_gen,
                               discriminator=discriminator, optimizer_disc=optimizer_disc)
    manager = tf.train.CheckpointManager(ckpt, directory=run_path + '/checkpoint',
                                         max_to_keep=2, keep_checkpoint_every_n_hours=4)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    while True:
        start = time.time()
        for images_a, weathers_a, images_b, weathers_b in dataset_train:

            if step % 10 == 0:
                print(epoch.numpy(), '-', step.numpy())

            maps = cutmix_maps(images_a.shape)
            gan_step(generator, optimizer_gen, discriminator, optimizer_disc,
                     images_a, weathers_a, images_b, weathers_b, maps,
                     summary_writer, step)

            if step % 100 == 0:
                weathers = tf.concat([weathers_a, weathers_b], axis=1)

                noise_1 = tf.random.normal([images_a.shape[0], noise_dim])
                generated_1 = generator([noise_1, images_a, weathers])
                noise_2 = tf.random.normal([images_a.shape[0], noise_dim])
                generated_2 = generator([noise_2, images_a, weathers])

                gen_l1 = tf.reduce_mean(tf.abs(images_b - generated_1))
                with summary_writer.as_default():
                    tf.summary.scalar('gen_l1', gen_l1, step=step)

                viz_img = tf.concat([images_a[0], images_b[0], generated_1[0], generated_2[0]], axis=1)

                _, discriminated_fake_1 = discriminator([images_a, weathers, generated_1])
                _, discriminated_fake_2 = discriminator([images_a, weathers, generated_2])
                _, discriminated_real = discriminator([images_a, weathers, images_b])
                logits = tf.concat([tf.zeros_like(discriminated_real[0]), discriminated_real[0],
                                    discriminated_fake_1[0], discriminated_fake_2[0]], axis=1)
                viz_labels = tf.tile(tf.tanh(logits), tf.constant([1, 1, 3], tf.int32))

                viz = tf.concat([viz_img, viz_labels], axis=0)
                image.write_png(viz, run_path + '/viz/' + str(epoch.numpy()) + '-' + str(step.numpy()) + '.png')

            step.assign_add(1)

        print('Time taken for epoch {} is {} sec\n'.format(epoch.numpy(), time.time() - start))
        epoch.assign_add(1)
        manager.save()
