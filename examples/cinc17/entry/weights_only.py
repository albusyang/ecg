
import tensorflow as tf
import sys

mpath = sys.argv[1]
model = tf.keras.models.load_model(mpath)
model.save_weights("model.hdf5")
