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

from keras import backend as K 


def get_model(input_shape, n_classes):

	inp_img = Input(shape=input_shape)

	if K.image_data_format() == 'channels_first':
		channel_axis = 1
	else:
		channel_axis = 3

	# Outer Conv2D 1
	X = Conv2D(32, (3, 3), activation='relu', name='outer_conv_1')(inp_img)


	# Residual Block 1
	X_1 = Conv2D(16, (1, 1), name='res_1_s_1x1')(X)
	X_1 = Activation('relu', name='res_1_act_1')(X_1)

	X_2 = Conv2D(32, (1, 1), name='res_1_e_1x1')(X_1)
	X_3 = Conv2D(32, (3, 3), padding='same', name='res_1_e_3x3')(X_1)

	X_f = Concatenate(axis=channel_axis, name='res_1_concat')([X_2, X_3])
	X_f = Conv2D(32, (1, 1), name='res_1_f_1x1')(X_f)
	X = Add(name='res_1_add')([X_f, X])
	X = BatchNormalization(name='batch_norm_1')(X)
	X = Activation('relu', name='res_1_act_f')(X)
	X = MaxPooling2D(pool_size=(2, 2), name='max_pool_res_1')(X)
	X = Dropout(0.5, name='drop_1')(X)

	# Outer Conv2D 2
	X = Conv2D(64, (1, 1), activation='relu', name='outer_conv_2')(X)

	# Residual Block 2
	X_1 = Conv2D(32, (1, 1), name='res_2_s_1x1')(X)
	X_1 = Activation('relu', name='res_2_act_1')(X_1)

	X_2 = Conv2D(64, (1, 1), name='res_2_e_1x1')(X_1)
	X_3 = Conv2D(64, (3, 3), padding='same', name='res_2_e_3x3')(X_1)

	X_f = Concatenate(axis=channel_axis, name='res_2_concat')([X_2, X_3])
	X_f = Conv2D(64, (1, 1), name='res_2_f_1x1')(X_f)
	X = Add(name='res_2_add')([X_f, X])
	X = BatchNormalization(name='batch_norm_2')(X)
	X = Activation('relu', name='res_2_act_f')(X)
	X = MaxPooling2D(pool_size=(2, 2), name='max_pool_res_2')(X)
	X = Dropout(0.5, name='drop_2')(X)

	# Outer Conv2D 3
	X = Conv2D(128, (1, 1), activation='relu', name='outer_conv_3')(X)

	# Residual Block 3
	X_1 = Conv2D(64, (1, 1), name='res_3_s_1x1')(X)
	X_1 = Activation('relu', name='res_3_act_1')(X_1)

	X_2 = Conv2D(128, (1, 1), name='res_3_e_1x1')(X_1)
	X_3 = Conv2D(128, (3, 3), padding='same', name='res_3_e_3x3')(X_1)

	X_f = Concatenate(axis=channel_axis, name='res_3_concat')([X_2, X_3])
	X_f = Conv2D(128, (1, 1), name='res_3_f_1x1')(X_f)
	X = Add(name='res_3_add')([X_f, X])
	X = BatchNormalization(name='batch_norm_3')(X)
	X = Activation('relu', name='res_3_act_f')(X)
	X = MaxPooling2D(pool_size=(2, 2), name='max_pool_res_3')(X)
	opt = Dropout(0.5, name='drop_3')(X)

	# Outer Conv2D 4
	X = Conv2D(256, (1, 1), activation='relu', name='outer_conv_4')(X)

	# Residual Block 4
	X_1 = Conv2D(128, (1, 1), name='res_4_s_1x1')(X)
	X_1 = Activation('relu', name='res_4_act_1')(X_1)

	X_2 = Conv2D(256, (1, 1), name='res_4_e_1x1')(X_1)
	X_3 = Conv2D(256, (3, 3), padding='same', name='res_4_e_3x3')(X_1)

	X_f = Concatenate(axis=channel_axis, name='res_4_concat')([X_2, X_3])
	X_f = Conv2D(256, (1, 1), name='res_4_f_1x1')(X_f)
	X = Add(name='res_4_add')([X_f, X])
	X = BatchNormalization(name='batch_norm_4')(X)
	X = Activation('relu', name='res_4_act_f')(X)
	X = MaxPooling2D(pool_size=(2, 2), name='max_pool_res_4')(X)
	X = Dropout(0.5, name='drop_4')(X)

	X = GlobalAveragePooling2D(name='global_avg_pooling_1')(X)

	X = Dense(units=1024, activation='relu', name='dense_1')(X)
	X = Dropout(0.5, name='drop_5')(X)
	X = Dense(units=512, activation='relu', name='dense_2')(X)
	X = Dropout(0.5, name='drop_6')(X)
	opt = Dense(n_classes, activation='softmax', name='dense_final')(X)
	
	model = Model(inp_img, opt)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	return model