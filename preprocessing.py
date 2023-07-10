from  sklearn.preprocessing import OneHotEncoder, LabelEncoder
import string
import os
import cv2
import numpy as np
import random


class Preprocessing():

    def __init__(self):

      all_characters = list(string.digits) + list(string.ascii_uppercase)

      self.n_characters = len(all_characters)

      label_encoder = LabelEncoder()

      encoded_characters = label_encoder.fit_transform(all_characters)

      self.label_to_index = {character:index for character, index in zip(all_characters, encoded_characters)}
      self.index_to_label = {index:character for character, index in zip(all_characters, encoded_characters)}

    def encode_text(self, text):

      encodes = []

      for character in text:

        encoded = np.zeros(self.n_characters)
        encoded[self.label_to_index[character]] = 1

        encodes.append(encoded)

      return np.array(encodes)

    def decode_text(self, indexes):

      return [self.index_to_label[index] for index in indexes]

def load_data(path):

    images = []
    labels = []

    for filename in os.listdir(path):

        if filename.endswith(".png"):

            img = cv2.imread(os.path.join(path, filename),0)

            # Convert source image to unsigned 8 bit integer Numpy array
            img = img / 255.0

            filename = filename.split(".")[0]

            images.append(img)
            labels.append(filename)

    return np.array(images), labels


preprocess = Preprocessing()

def get_data_generator(root_path, all_images, batch_size=16):
    images, labels = [], []
    while True:
      random.shuffle(all_images)
      for image_path in all_images:
          im = cv2.imread(os.path.join(root_path, image_path), 0)
          im = np.array(im) / 255.0
          images.append(np.array(im))
          labels.append(np.array(preprocess.encode_text(image_path.split(".")[0])))

          if len(images) >= batch_size:
              yield np.array(images), np.array(labels)
              images, labels = [], []