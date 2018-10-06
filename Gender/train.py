from preprocess import get_preprocessed_data
from temp_model import get_model 
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle

from keras.preprocessing.image import ImageDataGenerator


n_examples = -1
mode = 1
n_classes = 2
n_batches = 32
n_epochs = 30

X, Y = get_preprocessed_data(n_examples, mode, n_classes)

input_shape = X[0].shape 

model = get_model(input_shape, n_classes)
print(X.shape)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25)

datagen = ImageDataGenerator(rotation_range=30,
                            zoom_range=0.1,
                            vertical_flip=True,
                            fill_mode='nearest')

datagen.fit(X_train)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
filepath="models/model-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

history = model.fit_generator(datagen.flow(X, Y, batch_size=n_batches), 
                    validation_data=(X_val, Y_val), 
                    steps_per_epoch=X_train.shape[0] // n_batches, 
                    epochs=n_epochs, 
                    callbacks=callbacks_list)

with open('train_hisotry', 'wb') as f:
    pickle.dump(history.history, f)
