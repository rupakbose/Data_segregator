import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
from keras.layers import *
import tensorflow.keras.backend as K


def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))



def contrastive_loss(y, preds, margin=1):
	# explicitly cast the true class label data type to the predicted
	# class label data type (otherwise we run the risk of having two
	# separate data types, causing TensorFlow to error out)
	y = tf.cast(y, preds.dtype)
	# calculate the contrastive loss between the true labels and
	# the predicted labels
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	# return the computed contrastive loss to the calling function
	return loss


def gensample(image, label,model, times = 32 ):
  pos1 = []
  pos2 = []
  
  classes = np.random.randint(0,20,(times))
  for element in classes:
    idx = np.where(label==element)[0]
    np.random.shuffle(idx)
    pos1.append(image[idx[0]])
    pos2.append(image[idx[1]])
  neg1 = []
  neg2 = []
  dist = []

  for i in range(3):
    classes = np.random.randint(0,20,(times))
    for element in classes:
      idx1 = np.where(label ==element)[0]
      idx2 = np.where(label!=element)[0]
      np.random.shuffle(idx1)
      np.random.shuffle(idx2)
      neg1.append(image[idx1[0]])
      neg2.append(image[idx2[1]])

  neg1 = np.asarray(neg1)
  e1 = model.predict(AveragePooling2D()(neg1), batch_size = 32)      
  neg2 = np.asarray(neg2)
  e2 = model.predict(AveragePooling2D()(neg2), batch_size = 32)
  dis = np.sqrt(np.sum((e1 - e2)**2, axis = 1))
  id = np.where(dis<=0.6)[0]
  neg1 = neg1[id]
  neg2 = neg2[id]
  x1 = np.concatenate((pos1,neg1),axis = 0)
  x2 = np.concatenate((pos2,neg2),axis = 0)
  l = np.zeros(len(x1))
  l[:len(pos1)] = 1
  return x1,x2,l