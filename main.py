from preprocessing import *
import tensorflow as tf
from models import create_model
import config
import logging

images, labels = load_data(config.root_path)

preprocess = Preprocessing()
labels = np.array([preprocess.encode_text(label) for label in labels])


model = create_model(preprocess.n_characters)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
model.fit(images, labels, epochs=50)




