# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import tensorflow as tf
from keras import backend as K
from keras.layers import Activation, Add, Conv2D

from contrib.ops import SwitchNormalization


def hard_tanh(x):
    return K.clip(x, -1, )


def orthogonal(w):
    w_kw = K.int_shape(w)[0]
    w_kh = K.int_shape(w)[1]
    # w_w = K.int_shape(w)[2]
    # w_h = K.int_shape(w)[3]
    temp = 0
    for i in range(w_kw):
        for j in range(w_kh):
            wwt = tf.matmul(tf.transpose(w[i, j]), w[i, j])
            mi = K.ones_like(wwt) - K.identity(wwt)
            a = wwt * mi
            a = tf.matmul(tf.transpose(a), a)
            a = a * K.identity(a)
            temp += K.sum(a)
    return 2e-6 * temp


def residual_block(x, dim, ks, init_weight, name):
    y = Conv2D(dim, ks, strides=1, padding="same", kernel_initializer=init_weight, kernel_regularizer=orthogonal)(x)
    y = SwitchNormalization(axis=-1, name=name+'_0')(y)
    y = Activation('relu')(y)
    y = Conv2D(dim, ks, strides=1, padding="same", kernel_initializer=init_weight, kernel_regularizer=orthogonal)(y)
    y = SwitchNormalization(axis=-1, name=name+'_1')(y)
    return Add()([x, y])


def glu(x):
    channel = K.int_shape(x)[-1]
    channel = channel//2
    a = x[..., :channel]
    b = x[..., channel:]
    return a * K.sigmoid(b)
