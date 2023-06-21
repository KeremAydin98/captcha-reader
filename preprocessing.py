from  sklearn.preprocessing import OneHotEncoder
import string
import os
import cv2

class Preprocessing():

    def __init__(self):

        one_hot = OneHotEncoder()

        all_characters = [" "] + string.ascii_lowercase + string.ascii_uppercase + [0,1,2,3,4,5,6,7,8,9]

        onehot_characters = one_hot.fit_transform(all_characters)

        self.label_to_index = {character:index for character, index in zip(all_characters, onehot_characters)}
        self.index_to_label = {index:character for character, index in zip(all_characters, onehot_characters)}

    def encode_text(self, text):

        return [self.label_to_index[character] for character in text]

    def decode_text(self, indexes):

        return [self.index_to_label[index] for index in indexes]

def load_data(path):

    images = []
    labels = []

    for file in os.listdir(path):

        if file.endswith("png"):

            img = cv2.imread(os.path.join(path, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0

            label = file

            images.append(img)
            labels.append(label)

    return images, labels





