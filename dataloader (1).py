import tensorflow_datasets as tfds
import numpy as np


def gen():
  ds = tfds.load('uc_merced', split='train')

  image = []
  label = []
  for element in ds:
    if element.get('image').shape == (256,256,3):
      image.append(element.get('image'))
      label.append(element.get('label'))
      

  image = np.asarray(image)/255.0
  labels = np.asarray(label)

  return image, labels

if __name__ == '__main__':
  gen()



