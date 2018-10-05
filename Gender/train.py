from preprocess import get_preprocessed_data
from models import get_model 
from keras.callbacks import ModelCheckpoint
import pickle

n_examples = -1
mode = 1
n_classes = 2
n_batches = 32
n_epochs = 100

X, Y = get_preprocessed_data(-1, 1, 2)

input_shape = X[0].shape 

model = get_model(input_shape, n_classes)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


filepath="models/model-{epoch:02d}h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
history = model.fit(X, Y, validation_split=0.33, epochs=n_epochs, batch_size=n_batches, callbacks=callbacks_list, verbose=0)
with open('history_train', 'wb') as f:
  pickle.dump(history.history, f)
  
  
