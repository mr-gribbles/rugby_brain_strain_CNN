import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def build_brain_strain_cnn():
    model = Sequential()
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 10),
        strides=(1, 2),
        activation='relu',
        padding='valid',
        input_shape=(1, 3, 2000),
        data_format='channels_first'
    ))
    
    model.add(Conv2D(
        filters=32,
        kernel_size=(1, 10),
        strides=(1, 2),
        activation='relu',
        padding='valid',
        data_format='channels_first'
    ))
    
    model.add(Conv2D(
        filters=32,
        kernel_size=(1, 5),
        strides=(1, 1),
        activation='relu',
        padding='valid',
        data_format='channels_first'
    ))
    
    model.add(Flatten(data_format='channels_first'))
    model.add(Dropout(0.247))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=1e-6),
        metrics=['mse']
    )
    
    return model

model = build_brain_strain_cnn()
model.summary()
