from keras import backend as K 
from keras.layers import Conv2D, Activation
from keras.layers import MaxPooling2D
from keras.layers import Concatenate
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.models import Model



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


def get_squeeze_net(include_top=True, weights='imagenet', input_shape=None, pooling=None, classes=1000):

    if weights not in {'imagenet', None}:
        raise ValueError('The weights should be either `imagenet` or `None`')

    if weights == 'imagenet' and include_top and classes!=1000:
        raise ValueError('When weights selected is `imagenet` and `include_top` is `True`, classes should be 1000')

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=64,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    img_inp = Input(shape=input_shape)

    X = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv2d_1')(img_inp)

    X = MaxPooling2D(pool_size=(3, 3), strides=2, name='maxpool_1')(X)
    X = get_fire_module(X, filters=(16, 64, 64), layer_id=2)
    X = get_fire_module(X, filters=(16, 64, 64), layer_id=3)
    X = MaxPooling2D(pool_size=(3, 3), strides=2, name='maxpool_3')(X)

    X = get_fire_module(X, filters=(32, 128, 128), layer_id=4)
    X = get_fire_module(X, filters=(32, 128, 128), layer_id=5)
    X = MaxPooling2D(pool_size=(3, 3), strides=2, name='maxpool_5')(X)

    X = get_fire_module(X, filters=(48, 192, 192), layer_id=6)
    X = get_fire_module(X, filters=(48, 192, 192), layer_id=7)

    X = get_fire_module(X, filters=(64, 256, 256), layer_id=8)
    X = get_fire_module(X, filters=(64, 256, 256), layer_id=9)

    if include_top:
        weights_path = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"
        weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5', weights_path, cache_subdir='models')
        X = Dropout(0.5, name='dropout_9')(X)
        X = Conv2D(classes, (1, 1), name='conv_10')(X)
        X = Activation('relu')(X)
        X = GlobalAveragePooling2D()(X)
        final_opt = Activation('softmax', name='final_opt')(X)
    else:
        weights_path = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"
        weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5', weights_path, cache_subdir='models')
        if pooling == 'avg':
            final_opt = GlobalAveragePooling2D()(X)
        elif pooling == 'max':
            final_opt = GlobalMaxPooling2D()(X)
        elif pooling == None:
            pass
        else:
            raise ValueError('Unknown pooling option selected = ' + pooling)

    
    model = Model(inputs=img_inp, outputs=final_opt)

    model.load_weights(weights_path)

    return model
