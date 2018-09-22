from models import model
from utkface import UTKFace
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np 



class train(object):

    def __init__(self):
        pass


    def preprocess(self, n_examples, mode, test_size, n_classes):


        face = UTKFace.UTKFace()

        X, Y = face.load_data(n_examples, mode)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

        self.input_shape = X[0].shape 
        self.n_classes = n_classes

        Y_train = to_categorical(Y_train, self.n_classes)
        Y_test = to_categorical(Y_test, self.n_classes)

        print('X_train.shape : ' + str(X_train.shape))
        print('Y_train.shape : ' + str(Y_train.shape))
        print('X_test.shape : ' + str(X_test.shape))
        print('Y_test.shape : ' + str(Y_test.shape))

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        X_train /= 255.0
        X_test /= 255.0

        return (X_train, Y_train), (X_test, Y_test)



    def train(self, X, Y, n_conv, n_dense, d_format, model=None):
        
        batch_size = 32
        epochs = 10
        if model == None:
            model = model.model.get_model(self.input_shape, self.n_classes, n_conv, n_dense, d_format)
        
        model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

        model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

        model.save('model.h5')