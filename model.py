from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import adam
from keras import backend as K

class model(object):

    def __init__(self):
        pass

    
    def get_model(self, input_shape, n_classes, n_conv, n_dense, d_format):
        '''
            This function builds a convolutional neural network, with keras framework.

            @param1: input_shape: To pass as the input_shape to the first layer
            @param2: n_classes: To pass for the number of node in the last layer
            @param3: n_conv: No of total convolutiona layers

                The architecture of each convolutional layer is as follows
                Conv2D ---> BatchNormalization ---> Activation ---> MaxPooling2D --> Dropout

            @param4: n_dense: No of total dense layers

                The architecture of each dense layer is as follows
                Dense ---> Activation ---> Dropout

            @param5: d_format: Represents data_format, 'channels_first' or 'channels_last'
        '''

        model = Sequential()
        model.add(Conv2D(filters=64, 
                    kernel_size=(3, 3), 
                    padding='same', 
                    input_shape=input_shape,
                    data_format=d_format))

        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(rate=0.4))

        n_conv -= 1
        n_dense -= 1
        for i in range(n_conv):
            model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(rate=0.4))

        model.add(Flatten())

        for i in range(n_dense):
            model.add(Dense(units=1024))
            model.add(Activation('relu'))
            model.add(Dropout(rate=0.2))

        model.add(Dense(units=n_classes, activation='softmax'))

        return model
        
