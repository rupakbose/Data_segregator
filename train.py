import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
from keras.layers import *
import tensorflow.keras.backend as K
from model import base
import utils
import dataloader

import argparse

parser = argparse.ArgumentParser(' Parameters for training')
parser.add_argument("-s", "--savestep", required=True, type=int, default=10)
parser.add_argument("-e", "--epochs", required=True, type=int, default=100)
parser.add_argument("-lr", "--learning_rate", required=True, type=float, default=0.00005)
parser.add_argument("-emb", "--embedding", required=True, type=int, default=128)
args = parser.parse_args()

print('savestep is ', args.savestep)
print('epochs is ', args.epochs)
print('learning_rate is ', args.learning_rate)
print('embedding is ', args.embedding)

args = parser.parse_args()
base1 = base(args.embedding)


inputA = Input(shape=(None,None,3))
inputB = Input(shape=(None,None,3))
featA = base1(inputA)
featB = base1(inputB)
dist = utils.euclidean_distance([featA,featB])
siam = tf.keras.Model((inputA,inputB), dist)
siam.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), loss = utils.contrastive_loss)

print('loading data')
image, labels = dataloader.gen()
print('image samples total are ', len(image))

for i in range(args.epochs):
  x1,x2,l = utils.gensample(image, labels, base1)
  print('% samples under cut off', np.round(len(np.where(l==0)[0])/ len(l),4))
  x1 = AveragePooling2D()(x1)
  x2 = AveragePooling2D()(x2)
  siam.fit((x1,x2), l, batch_size= 32, verbose =1)
  if i % args.savestep == 0:
    base1.save('model.h5')
