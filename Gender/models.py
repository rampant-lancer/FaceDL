from FireModule import get_fire_module
from keras.layers import Add
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Model

from keras.utils import plot_model

def get_res_fire(X, filters, p_filter, layer_id):
    X = Conv2D(filters=p_filter, kernel_size=(1, 1), name='temp_'+str(layer_id))(X)
    X_1 = get_fire_module(X, filters, layer_id)
    print(X.shape)
    X_1 = Activation('relu')(X_1)
    X_2 = get_fire_module(X_1, filters, layer_id+1)
    print(X_2.shape)
    X = Add()([X, X_2])
    X = Activation('relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling2D(pool_size=(2, 2), name='maxpool_'+str(layer_id))(X)
    X = Dropout(0.5)(X)
    return X



def get_model(input_shape, n_classes):

    inp_img = Input(shape=input_shape)

    X = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='conv_1')(inp_img)

    X = get_res_fire(X, (16, 32), 32, 2)

    X = get_res_fire(X, (32, 64), 64, 4)
    
    X = get_res_fire(X, (64, 128), 128, 6)

    X = get_res_fire(X, (128, 256), 256, 8)

    
    X = GlobalAveragePooling2D()(X)

    X = Dense(units=1024, activation='relu', name='dense_9')(X)
    X = Dropout(0.5)(X)
    opt = Dense(units=n_classes, activation='softmax', name='final_dense')(X)

    model = Model(inp_img, opt)
    plot_model(model, to_file='model.png')
    print(model.summary())
    
    return model
