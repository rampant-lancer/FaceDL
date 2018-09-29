from utkface import UTKFace
from keras import backend as K 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.layers import Concatenate
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Model
from keras import optimizers

squeeze_1 = '_squeeze1x1'
expand_1 = '_expand1x1'
expand_3 = '_expand3x3'


def get_fire_module(X, filters, layer_id):
    layer_name = 'fire_' + str(layer_id)

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    X = Conv2D(filters=filters[0], kernel_size=(1, 1), activation='relu', name=layer_name+squeeze_1)(X)

    X_1 = Conv2D(filters=filters[1], kernel_size=(1, 1), activation='relu', name=layer_name+expand_1)(X)
    X_3 = Conv2D(filters=filters[2], kernel_size=(3, 3), activation='relu', padding='same', name=layer_name+expand_3)(X)

    return Concatenate(axis=channel_axis, name='concatenate_' + str(layer_id))([X_1, X_3])


def preprocess(n_examples, mode, n_classes):
    face = UTKFace.UTKFace()

    X, Y = face.load_data(n_examples, mode)
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

    n_classes = n_classes

    Y = to_categorical(Y, n_classes)


    print('X.shape : ' + str(X.shape) + ' Y.shape : ' + str(Y.shape))
  
    X = X.astype(np.float16)


    X /= 255.0

    return X, Y

def get_model(input_shape, n_classes):
    img_inp = Input(input_shape=input_shape)

    X = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv2d_1')(img_inp)

    X = MaxPooling2D(pool_size=(3, 3), strides=2, name='maxpool_1')(X)
    X = get_fire_module(X, filters=(16, 64, 64), layer_id=2)
    X = get_fire_module(X, filters=(16, 64, 64), layer_id=3)
    X = MaxPooling2D(pool_size=(3, 3), strides=2, name='maxpool_3')(X)

    X = get_fire_module(X, filters=(48, 192, 192), layer_id=6)
    X = get_fire_module(X, filters=(48, 192, 192), layer_id=7)

    X = Flatten()(X)
    X = Dense(units=1024, activation='relu', name='dense_1')(X)
    X = Dropout(0.5, name='drop_1')(X)
    output = Dense(units=n_classes, activation='softmax')(X)

    model = Model(inputs=img_inp, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    return model



X, Y = preprocess(-1, 1, 2)
input_shape = X[0].shape
n_classes = 2
model = get_model(input_shape, n_classes)
batch_size = 64
epochs = 10

import pickle
history = model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

with open('train_history', 'wb') as f:
    pickle.dump(history.history, f)

model.save('Model_fire.h5')





