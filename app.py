import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras
from sklearn.cluster import DBSCAN
import io

st.write('Hello')

model = keras.models.load_model('model.h5')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  bytes_data = Image.open(uploaded_file)
  demo = np.asarray(bytes_data)/255.0
  print(demo.shape)

if uploaded_file is not None:
  patchsize = int(st.number_input('Insert a patch_size', min_value=24))

if uploaded_file is not None and patchsize >= 24:
  demo1 = demo[:demo.shape[0]//patchsize*patchsize,:demo.shape[1]//patchsize*patchsize,: ]
  im = []
  for i in range(0,demo1.shape[0],patchsize):
    for j in range(0,demo1.shape[1],patchsize):
      im.append(demo1[i:i+patchsize,j:j+patchsize,:])

  emb = model.predict(np.asarray(im), batch_size=32)

  l=[]
  p = []
  for i in range(1,10):
    print(i)
    db_default = DBSCAN(eps = i/10, min_samples = 100).fit(emb)
    labels_db = db_default.labels_
    l.append(len(np.unique(labels_db)))
    p.append(len(np.where(labels_db==-1)[0])/len(labels_db))
    # print(np.unique(labels_db), len(np.where(labels_db==-1)[0])/len(labels_db))

  index = np.where(l==max(np.asarray(l)))[0]
  final = index[np.argmax(np.asarray(p)[index])]

  db_default = DBSCAN(eps = (final+1)/10, min_samples = 100).fit(emb)
  labels_db = db_default.labels_
  labels_db = (labels_db - np.amin(labels_db))/(np.amax(labels_db) - np.amin(labels_db))

  asdf = np.zeros(demo1.shape)
  m = 0
  for i in range(0,demo1.shape[0],patchsize):
    for j in range(0,demo1.shape[1],patchsize):
      asdf[i:i+patchsize,j:j+patchsize,:] = asdf[i:i+patchsize,j:j+patchsize,:] + labels_db[m]
      m+=1
  
  asdf = Image.fromarray(np.uint8(asdf*255))

  st.image(asdf, caption='patches', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  st.image(demo1, caption='image', width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
  im = np.asarray(im)
  for element in np.unique(labels_db):
    with io.BytesIO() as buffer:
      # Write array to buffer
        np.save(buffer, im[np.where(labels_db==element)[0]])
        strng = "Download numpy array (.npy)_" + str(element) + "_ as label"
        btn = st.download_button(
            label=strng,
            data = buffer, # Download buffer
            file_name = 'predicted_map.npy'
        ) 

