from utkface import UTKFace
from keras.utils import to_categorical
import numpy as np


def get_preprocessed_data(n_examples, mode, n_classes):
    face = UTKFace.UTKFace()
    X, Y = face.load_data(n_examples, mode)
    X = X.astype(np.float16) / 255.0
    Y = to_categorical(Y, num_classes=n_classes)

    return X, Y




    

