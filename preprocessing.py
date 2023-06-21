from  sklearn.preprocessing import OneHotEncoder, LabelEncoder
import string
import os
import cv2
import numpy as np

class Preprocessing():

    def __init__(self):

        all_characters = [" "] + list(string.ascii_lowercase) + list(string.ascii_uppercase) + list(string.digits)

        self.n_characters = len(all_characters)

        label_encoder = LabelEncoder()

        encoded_characters = label_encoder.fit_transform(all_characters)

        self.label_to_index = {character:index for character, index in zip(all_characters, encoded_characters)}
        self.index_to_label = {index:character for character, index in zip(all_characters, encoded_characters)}

    def encode_text(self, text):

        return np.array([self.label_to_index[character] for character in text])

    def decode_text(self, indexes):

        return [self.index_to_label[index] for index in indexes]

def load_data(path):

    images = []
    labels = []

    for file in os.listdir(path):

        if file.endswith(".png"):

            img = cv2.imread(os.path.join(path, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (200, 100))
            img = img / 255.0

            file = file.split(".")[0]

            if len(file) < 10:

                label = file + (10 - len(file)) * " "

            else:

                label = file

            images.append(img)
            labels.append(label)

    return np.array(images), labels





