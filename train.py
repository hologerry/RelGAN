# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import argparse
# import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from relgan import Relgan

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, default='data/celeba-hq/celeba-256', help="data path")
parser.add_argument("-d", "--device", type=str, default='0', help="gpu device")
parser.add_argument("-g", "--growth", type=bool, default=False, help="allow_growth")
parser.add_argument("-s", "--step", type=int, default=0, help="train_step")
parser.add_argument("-l", "--lr", type=float, default=5e-5)
parser.add_argument("-b1", "--beta1", type=float, default=0.5)
parser.add_argument("-b2", "--beta2", type=float, default=0.999)
parser.add_argument("-batch", "--batch_size", type=int, default=4)
parser.add_argument("-sample", "--sample_size", type=int, default=2)
parser.add_argument("-ep", "--epochs", type=int, default=400000)  # number of iterations
parser.add_argument("-l1", "--lambda1", type=int, default=10)
parser.add_argument("-l2", "--lambda2", type=int, default=10)
parser.add_argument("-l4", "--lambda4", type=int, default=10)
parser.add_argument("-l5", "--lambda5", type=int, default=10)
parser.add_argument("-gp", "--lambda_gp", type=int, default=150)
parser.add_argument("-img", "--img_size", type=int, default=256)
parser.add_argument("-v", "--vec_size", type=int, default=17)

args = parser.parse_args()


# os.environ["CUDA_VISIBLE_DEVICES"] = args.device


# K.set_floatx('float64')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

rel_gan = Relgan(args)
rel_gan.train()
