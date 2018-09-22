from utkface import UTKFace
from keras import backend as K 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np




def get_model(input_shape, n_classes):
    K.set_image_data_format('channels_last')
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.4))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.4))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.4))


    model.add(Flatten())

    model.add(Dense(units=2048))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(rate=0.4))

    model.add(Dense(units=1028))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(rate=0.4))

    model.add(Dense(units=n_classes, activation='softmax'))

    return model


def preprocess(n_examples, mode, n_classes):
    face = UTKFace.UTKFace()

    X, Y = face.load_data(n_examples, mode)
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

    input_shape = X[0].shape 
    n_classes = n_classes

    #Y_train = to_categorical(Y_train, n_classes)
    #Y_test = to_categorical(Y_test, n_classes)

    Y = to_categorical(Y, n_classes)

    #print('X_train.shape : ' + str(X_train.shape))
    #print('Y_train.shape : ' + str(Y_train.shape))
    #print('X_test.shape : ' + str(X_test.shape))
    #print('Y_test.shape : ' + str(Y_test.shape))

    print('X.shape : ' + str(X.shape) + ' Y.shape : ' + str(Y.shape))
    #X_train = X_train.astype(np.float32)
    #X_test = X_test.astype(np.float32)

    X = X.astype(np.float16)

    #X_train /= 255.0
    #X_test /= 255.0

    X /= 255.0

    return X, Y


def train():
    n_classes = 2
    mode = 1
    n_examples = -1
    batch_size = 64
    epochs = 10

    X_train, Y_train = preprocess(n_examples, mode, n_classes)

    input_shape = X_train[0].shape

    model = get_model(input_shape, n_classes)
    print(model.summary())
    
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    model.save('Model.h5')

train()
