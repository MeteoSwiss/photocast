# Copyright (c) 2022 MeteoSwiss, contributors listed in AUTHORS
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

import math

import tensorflow as tf
from tensorflow.keras.layers import Layer

from .normalization import SpectralNormalization


class SelfAttentionAxis(Layer):
    """
    Arguments:
        heads: The number of attention heads
        channels_reduced: The number of channels per head for the query, key and value vectors.
        axis: The axis along which self-attention is performed, either 'row' or 'column'
    """
    def __init__(self, heads, channels_reduced, axis='column', **kwargs):
        super().__init__(**kwargs)

        self.heads = heads
        self.channels_reduced = channels_reduced
        self.axis = axis

        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None

        self.gamma = None

    def build(self, input_shape):
        super().build(input_shape)
        channels_in = input_shape[3]

        # trainable channel weight matrices, perform per-head 1x1 convolutions of the input layer
        self.W_q = self.add_weight(
            name='W_q', shape=(self.heads, channels_in, self.channels_reduced), trainable=True)
        self.W_k = self.add_weight(
            name='W_k', shape=(self.heads, channels_in, self.channels_reduced), trainable=True)
        self.W_v = self.add_weight(
            name='W_v', shape=(self.heads, channels_in, self.channels_reduced), trainable=True)
        self.W_o = self.add_weight(
            name='W_o', shape=(self.heads * self.channels_reduced, channels_in), trainable=True)

        # weighting of input and self-attention layers
        self.gamma = self.add_weight(name='gamma', shape=(1,), initializer='zero', trainable=True)

    def call(self, x, **kwargs):
        if self.axis == 'row':
            x = tf.transpose(x, [0, 2, 1, 3])  # flip rows and columns

        x_shape = x.shape
        rows = x_shape[1]
        cols = x_shape[2]

        Q = tf.einsum('bwli,hir->bhwlr', x, self.W_q)  # query matrix (bs, heads, rows, cols, channels_reduced)
        K = tf.einsum('bwli,hir->bhwlr', x, self.W_k)  # key matrix
        V = tf.einsum('bwli,hir->bhwlr', x, self.W_v)  # value matrix

        S = tf.einsum('bhwlr,bhwmr->bhwlm', Q, K)  # dot-product similarity of Q and K (bs, heads, rows, cols, cols)
        S = S/math.sqrt(self.channels_reduced)
        S = tf.nn.softmax(S, axis=4)
        A = tf.einsum('bhwlm,bhwmr->bhwlr', S, V)  # attention (bs, heads, rows, cols, channels_reduced)

        A_concat = tf.transpose(A, [0, 2, 3, 1, 4])  # (bs, rows, cols, heads, channels_reduced)
        A_concat = tf.reshape(A_concat, [-1, rows, cols, self.heads * self.channels_reduced])

        O = tf.einsum('bwlt,ti->bwli', A_concat, self.W_o)  # (bs, rows, cols, channels_in)

        y = self.gamma*O + x

        if self.axis == 'row':
            y = tf.transpose(y, [0, 2, 1, 3])  # flip back rows and columns

        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class SelfAttention2D(Layer):
    """
    Arguments:
        heads: The number of attention heads
        channels_reduced: The number of channels per head for the query, key and value vectors.
        constraint: The constraint on the channel projection weight matrices, either None or 'specnorm'
    """
    def __init__(self, heads, channels_reduced, constraint=None, **kwargs):
        super().__init__(**kwargs)

        self.heads = heads
        self.channels_reduced = channels_reduced
        self.constraint = constraint

        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None

        self.gamma = None

    def build(self, input_shape):
        super().build(input_shape)
        channels_in = input_shape[3]

        # trainable channel weight matrices, perform per-head 1x1 convolutions of the input layer
        self.W_q = self.add_weight(
            name='W_q', shape=(self.heads, channels_in, self.channels_reduced),
            constraint=SpectralNormalization(), trainable=True)
        self.W_k = self.add_weight(
            name='W_k', shape=(self.heads, channels_in, self.channels_reduced),
            constraint=SpectralNormalization(), trainable=True)
        self.W_v = self.add_weight(
            name='W_v', shape=(self.heads, channels_in, self.channels_reduced),
            constraint=SpectralNormalization(), trainable=True)
        self.W_o = self.add_weight(
            name='W_o', shape=(self.heads * self.channels_reduced, channels_in),
            constraint=SpectralNormalization(), trainable=True)

        # weighting of input and self-attention layers
        self.gamma = self.add_weight(name='gamma', shape=(1,), initializer='zero', trainable=True)

    def call(self, x, **kwargs):
        x_shape = x.shape
        rows = x_shape[1]
        cols = x_shape[2]
        channels_in = x_shape[3]

        X_ = tf.reshape(x, [-1, rows*cols, channels_in])

        Q = tf.einsum('bni,hir->bhnr', X_, self.W_q)  # query matrix (bs, heads, rows*cols, channels_reduced)
        K = tf.einsum('bni,hir->bhnr', X_, self.W_k)  # key matrix
        V = tf.einsum('bni,hir->bhnr', X_, self.W_v)  # value matrix

        S = tf.einsum('bhnr,bhmr->bhnm', Q, K)  # dot-product similarity of Q and K (bs, heads, rows*cols, rows*cols)
        S = S/math.sqrt(self.channels_reduced)
        S = tf.nn.softmax(S, axis=3)
        A = tf.einsum('bhnm,bhmr->bhnr', S, V)  # attention (bs, heads, rows*cols, channels_reduced)

        A_concat = tf.transpose(A, [0, 2, 1, 3])  # (bs, rows*cols, heads, channels_reduced)
        A_concat = tf.reshape(A_concat, [-1, rows * cols, self.heads * self.channels_reduced])

        O = tf.einsum('bnt,ti->bni', A_concat, self.W_o)  # (bs, rows*cols, channels_in)
        O = tf.reshape(O, [-1, rows, cols, channels_in])

        y = self.gamma*O + x
        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class SelfAttention2D_old(Layer):

    def __init__(self, reduced_filters, **kwargs):
        super().__init__(**kwargs)
        self.reduced_filters = reduced_filters

    def build(self, input_shape):
        super().build(input_shape)
        self.w_f = self.add_weight(name='w_f', shape=(1, 1, input_shape[-1], self.reduced_filters),
                                   constraint=SpectralNormalization(), trainable=True)
        self.w_g = self.add_weight(name='w_g', shape=(1, 1, input_shape[-1], self.reduced_filters),
                                   constraint=SpectralNormalization(), trainable=True)
        self.w_h = self.add_weight(name='w_h', shape=(1, 1, input_shape[-1], self.reduced_filters),
                                   constraint=SpectralNormalization(), trainable=True)
        self.w_v = self.add_weight(name='w_v', shape=(1, 1, self.reduced_filters, input_shape[-1]),
                                   constraint=SpectralNormalization(), trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(1,), initializer='zero', trainable=True)

    def call(self, x, **kwargs):
        x_shape = tf.shape(x)

        x_f = tf.keras.backend.conv2d(x, self.w_f, padding="same")
        x_g = tf.keras.backend.conv2d(x, self.w_g, padding="same")
        x_h = tf.keras.backend.conv2d(x, self.w_h, padding="same")

        x_f = tf.reshape(x_f, (x_shape[0], x_shape[1] * x_shape[2], -1))
        x_g = tf.reshape(x_g, (x_shape[0], x_shape[1] * x_shape[2], -1))
        x_h = tf.reshape(x_h, (x_shape[0], x_shape[1] * x_shape[2], -1))

        y = tf.matmul(x_f, x_g, transpose_b=True)  # attention map
        y = tf.nn.softmax(y, axis=1)
        y = tf.matmul(y, x_h)
        y = tf.reshape(y, (x_shape[0], x_shape[1], x_shape[2], -1))
        y = tf.keras.backend.conv2d(y, self.w_v, padding="same")
        y = self.gamma*y + x
        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class SelfAttentionAxis_old(Layer):

    def __init__(self, reduced_filters, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.reduced_filters = reduced_filters
        self.axis = axis

    def build(self, input_shape):
        super().build(input_shape)
        self.w_f = self.add_weight(name='w_f', shape=(1, 1, input_shape[-1], self.reduced_filters),
                                   constraint=SpectralNormalization(), trainable=True)
        self.w_g = self.add_weight(name='w_g', shape=(1, 1, input_shape[-1], self.reduced_filters),
                                   constraint=SpectralNormalization(), trainable=True)
        self.w_h = self.add_weight(name='w_h', shape=(1, 1, input_shape[-1], self.reduced_filters),
                                   constraint=SpectralNormalization(), trainable=True)
        self.w_v = self.add_weight(name='w_v', shape=(1, 1, self.reduced_filters, input_shape[-1]),
                                   constraint=SpectralNormalization(), trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(1,), initializer='zero', trainable=True)

    def call(self, x, **kwargs):
        x_shape = tf.shape(x)

        x_f = tf.keras.backend.conv2d(x, self.w_f, padding="same")
        x_g = tf.keras.backend.conv2d(x, self.w_g, padding="same")
        x_h = tf.keras.backend.conv2d(x, self.w_h, padding="same")

        if self.axis == 2:
            x_f = tf.transpose(x_f, [0, 2, 1, 3])
            x_g = tf.transpose(x_g, [0, 2, 1, 3])
            x_h = tf.transpose(x_h, [0, 2, 1, 3])

        x_f = tf.reshape(x_f, (x_shape[0] * x_shape[1], x_shape[2], -1))
        x_g = tf.reshape(x_g, (x_shape[0] * x_shape[1], x_shape[2], -1))
        x_h = tf.reshape(x_h, (x_shape[0] * x_shape[1], x_shape[2], -1))

        y = tf.matmul(x_f, x_g, transpose_b=True)  # attention map
        y = tf.nn.softmax(y, axis=1)
        y = tf.matmul(y, x_h)
        y = tf.reshape(y, (x_shape[0], x_shape[1], x_shape[2], -1))

        # if self.axis == 2:
        #     y = tf.transpose(y, [0, 2, 1, 3])

        y = tf.keras.backend.conv2d(y, self.w_v, padding="same")
        y = self.gamma*y + x

        return y

    def compute_output_shape(self, input_shape):
        return input_shape
