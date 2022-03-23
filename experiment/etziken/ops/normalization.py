# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.linalg import matvec
from tensorflow.nn import l2_normalize


class SpectralNormalization(Constraint):
    def __init__(self, iterations=1):
        self.iterations = iterations
        self.u = None

    def __call__(self, w):
        output_neurons = w.shape[-1]
        W_ = tf.reshape(w, [-1, output_neurons])
        if self.u is None:
            self.u = tf.Variable(
                initial_value=tf.random_normal_initializer()(shape=(output_neurons, ), dtype='float32'),
                trainable=False)

        u_ = self.u
        v_ = None
        for _ in range(self.iterations):
            v_ = matvec(W_, u_)
            v_ = l2_normalize(v_)

            u_ = matvec(W_, v_, transpose_a=True)
            u_ = l2_normalize(u_)

        sigma = tf.tensordot(u_, matvec(W_, v_, transpose_a=True), axes=1)
        self.u.assign(u_)  # '=' produces an error in graph mode
        return w / sigma
