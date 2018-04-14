"""
The neural network was inspired by VGG-16. 
The database is not included in this repo, please download the database from Kaggle
(https://www.kaggle.com/c/digit-recongnizer) and change into a csv format. The first
row should be moved from raw data. The first column of raw data is label.
Use TensorFlow backend.
"""
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import keras
import csv
import numpy as np
import tensorflow as tf

np.random.seed(123)
DataPath = './data/Data_mnist.csv'
LabelPath = './data/Label_mnist.csv'
class_num = 10


def read_csv(filename):
    if filename:
        with open(filename, 'rU') as f:
            data = csv.reader(f)
            data = list(data)
            try:
                data = np.array(data[:], dtype=float)
            except ValueError as e:
                print("Error while puttin csv data into numpy array")
                print("ValueError: {}".format(e))
                raise ValueError
            return data[:, :]
    else:
        raise IOError('Non-empty filename expected.')


def transform_data(data, feature_dim=28, features_number=28):
    data_stacked = np.array([r.reshape(features_number, feature_dim).transpose() for r in data])
    return data_stacked


def kaggle_cnn():
    inp = Input(shape=(28, 28, 1))
    out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal',
                 data_format='channels_last')(inp)
    out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal',
                 data_format='channels_last')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(out)
    out = Dropout(0.2)(out)
    out = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', kernel_initializer='he_normal',
                 padding='same', data_format='channels_last')(out)
    out = Conv2D(filters=64, kernel_size=(1, 3), activation='relu', kernel_initializer='he_normal',
                 padding='same', data_format='channels_last')(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal',
                 padding='same', data_format='channels_last')(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(out)
    out = Dropout(0.25)(out)
    out = Conv2D(filters=128, kernel_size=(3, 1), activation='relu', kernel_initializer='he_normal',
                 padding='same', data_format='channels_last')(out)
    out = Conv2D(filters=128, kernel_size=(1, 3), activation='relu', kernel_initializer='he_normal',
                 padding='same', data_format='channels_last')(out)
    out = BatchNormalization()(out)
    out = Dropout(0.25)(out)
    out = Flatten()(out)
    out = Dense(128, activation='relu')(out)
    out = BatchNormalization()(out)
    out = Dropout(0.25)(out)
    out = Dense(class_num, activation='softmax')(out)
    model = Model(inputs=inp, outputs=out)

    return model


# Main function
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

Data = read_csv(DataPath)
Label = read_csv(LabelPath)
Data = Data / 255.0 # Data Normalization
Data = transform_data(Data)
Data = np.expand_dims(Data, axis=3)
Label = keras.utils.to_categorical(Label, num_classes=class_num)
x_train, x_val, y_train, y_val = train_test_split(Data, Label, test_size=0.2, random_state=123)

callbacks = [TensorBoard('./logs', histogram_freq=1),
             ModelCheckpoint('weights-best-kaggle.hdf5', monitor='val_loss', save_best_only=True, verbose=1),
             ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.0001)]

model = kaggle_cnn()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
plot_model(model, to_file='model.png')
model.fit(x_train, y_train, batch_size=64, epochs=40, verbose=2, validation_data=(x_val, y_val), callbacks=callbacks)
score = model.evaluate(x_val, y_val, verbose=0)
print 'val_loss:', score[0]
print 'val_accuracy:', score[1]
