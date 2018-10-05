"""
FireModule implemented in Keras

This implementation is the building block of the squeeze nets from the following paper:
@online{
    1602.07360,
    Author = {Forrest N. Iandola and Song Han and Matthew W. Moskewicz and Khalid Ashraf and William J. Dally and Kurt Keutzer},
    Title = {SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size},
    Year = {2016},
    Eprint = {1602.07360},
    Eprinttype = {arXiv},
}

@author: Sarfarazul Haque
"""


import keras.backend as K 
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Concatenate
import numpy as np 

def get_fire_module(X, filters, layer_id):
    '''
        @param1: X: Input tensor to the fire module block.
        @param2: filters: Single integer or tuple of integers representing the 
                            number of filters in each Convolutional Layer.
        @param3: layer_id: Ints, To provide the distinctive name to the layers.

        @return: resultant tensor after applying fire module to the input tensor.
    '''

    NO_OF_CONV_LAYERS = 3
    _filters = np.zeros(NO_OF_CONV_LAYERS, dtype=np.uint16)
    len_filters = len(filters)


    assert (len_filters > 0), 'Fitlers size should be `Non-zero`'
    assert (len_filters <= NO_OF_CONV_LAYERS), 'Size of filters array should be less then `3`'

    ''' To set the number of filters in each of the Convolutional Layer '''
    if len_filters == 1:    # If provided argument has single value.
        for i in range(NO_OF_CONV_LAYERS):  # Then No of Conv-filters in each Conv-layers is same
            _filters[i] = filters
    elif len_filters == 2:  # If provided argument has two values.                    
        _filters[0] = filters[0]    # Then 0th element will represent the No of Conv-filters in 0th Conv-layer  
        for i in range(NO_OF_CONV_LAYERS - 1):  # and remaining value will represent 
            _filters[i] = filters[1]            # No of Conv-filters in remaining Conv-layers
    else:   # If the provided argument has three values.
        _filters = np.array(filters, dtype=np.uint16) # Then each value will represnet No of Conv-filters in corresponding Conv-layer

    # Setting layer name
    layer_name = 'fire_' + str(layer_id)

    # Checking for the axis about which concatenation takes place
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    squeeze_1 = '_squeeze_1x1'
    expand_1 = '_expand_1x1'
    expand_3 = '_expand_3x3'

    ''' Building Fire Module '''

    # Squeezed Part of Fire Module
    X = Conv2D(filters=_filters[0], kernel_size=(1, 1), activation='relu', name=layer_name+squeeze_1)(X)

    # Expanded Part of Fire Module
    X_1 = Conv2D(filters=_filters[1], kernel_size=(1, 1), name=layer_name+expand_1)(X)
    X_3 = Conv2D(filters=_filters[2], kernel_size=(3, 3), padding='same', name=layer_name+expand_3)(X)

    # Concatenating and returning the resultant tensor
    X_final = Concatenate(axis=channel_axis, name='concatenate_' + str(layer_id))([X_1, X_3])

    return X_final
