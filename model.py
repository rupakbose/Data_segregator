import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
from keras.layers import *
import tensorflow.keras.backend as K
import utils

# base model
def base(dim = 128):
  input = Input(shape=(None,None,3))
  mid = Conv2D(32,4,2,activation='relu', padding='same')(input)
  mid = Conv2D(64,4,2,activation='relu', padding='same')(mid)
  mid = Conv2D(dim//2,4,2,activation='relu', padding='same')(mid)
  a = GlobalAveragePooling2D()(mid)
  b = GlobalMaxPooling2D()(mid)
  c = concatenate([a,b])
  return tf.keras.Model(input,c)

