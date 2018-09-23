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
from keras.models import Model


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


n_examples = -1
mode = 1
n_classes = 2

X, Y = preprocess(n_examples, mode, n_examples)

input_shape = X[0].shape

from keras.applications.xception import Xception

# Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


model = Xception(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max')

model_opt = model.output
model_opt = Dense(1024, activation='relu')(model_opt)
model_opt = Dropout(0.5)(model_opt)
model_opt = Dense(1024, activation='relu')(model_opt)
predictions = Dense(n_classes, activation='softmax')(model_opt)


model_final = Model(input = model.input, output = predictions)

for layer in model_final.layers[:]:
    layer.trainable = False

for layer in model_final.layers[-5:]:
    layer.trainable = True

model_final.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 64
epochs = 10
history = model_final.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

model.save('Model_2_Xception.h5')