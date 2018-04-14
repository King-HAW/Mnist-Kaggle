"""
The neural network was inspired by VGG-16. 
The database is not included in this repo, please download the database from Kaggle
(https://www.kaggle.com/c/digit-recongnizer) and change into a csv format. The first
row should be moved from raw data. The first column of raw data is label.
Use TensorFlow backend.
"""
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout
import keras
import csv
import numpy as np
import tensorflow as tf

np.random.seed(123)
# DataPath = './data/Data_mnist.csv'
# LabelPath = './data/Label_mnist.csv'
TestPath = './data/Data_mnist_test.csv'
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


def save_csv(filename, data):
    data = [(i+1, r) for i, r in enumerate(data)]
    data.insert(0, ('ImageId', 'Label'))
    with open(filename, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(data)


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
Data = read_csv(TestPath)
Data = Data / 255.0 # Data Normalization
Data = transform_data(Data)
Data = np.expand_dims(Data, axis=3)

model = kaggle_cnn()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('weights-best-kaggle.hdf5')
result = model.predict(Data)
result = np.argmax(result, axis=1)
csv_out_file = './data/output_predictions_kaggle.csv'
save_csv(csv_out_file, result)
print 'Success!'
