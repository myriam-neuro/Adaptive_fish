'''

@author Myriam Hamon

'''
##
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tqdm import tqdm
import pandas as pd
import os
import sys
import glob

def training (outpath,name, directory_0, directory_1, data_points =  9000, sequence_size = 150, continuous_recording = 'True'):
    '''

    :param outpath: directory where the model will be saved
    :param name : name of the model
    :param directory_0: directory of the files with real fish data (type csv with 4 columns, deliminator ;)
    :param directory_1: directory of the files with fake fish data (type csv with 4 columns, deliminator ;)
    :param data_points: number of datapoints taken from the csv files, has to be a multiple of sequence_size
    :param sequence_size: size of the sequence given to the discriminator
    :param continuous_recording: whether the recording of the data is continuous (without NaNs) or not, default = True
    :return: save a model in the outpath with the name
    '''

    lstm_cells = 150
    input_nodes = 4

    # load data
    file_names_fake = glob.glob(os.path.join(directory_0, "*.csv"))
    file_names_real  = glob.glob(os.path.join(directory_1, "*.csv"))
    fish_complete_1 = np.zeros((len(file_names_fake), data_points, 2))
    fish_complete_2 = np.zeros((len(file_names_fake), data_points, 2))
    fish_complete_3 = np.zeros((len(file_names_real), data_points, 2))
    fish_complete_4 = np.zeros((len(file_names_real), data_points, 2))


    for i in tqdm(range(len(file_names_fake))):
        trajectory = pd.read_csv(file_names_fake[i], delimiter=';')[0:data_points].T
        fish_complete_1[i] = trajectory[0:2].T
        fish_complete_2[i] = trajectory[2:4].T
    Training_data_fake = sequence(fish_complete_1,fish_complete_2,data_points,sequence_size,continuous_recording=continuous_recording)

    for i in tqdm(range(len(file_names_real))):
        trajectory = pd.read_csv(file_names_fake[i], delimiter=';')[0:data_points].T
        fish_complete_3[i] = trajectory[ 0:2].T
        fish_complete_4[i] = trajectory[2:4].T
    Training_data_real = sequence(fish_complete_1, fish_complete_2, data_points, sequence_size,
                                       continuous_recording=continuous_recording)

    #label data
    Training_labels_real = np.ones(Training_data_real.shape[0])
    Training_labels_fake = np.zeros (Training_data_fake.shape[0])

    Training_data = np.concatenate((Training_data_real, Training_data_fake))
    Training_labels = np.concatenate((Training_labels_real, Training_labels_fake))

    # shuffle and rotate data
    Training_data_processed,Training_labels_processed = transform_shuffle(Training_data, Training_labels, partner='True')


    #Build model

    lstm_model = models.Sequential()
    lstm_model.add(layers.LSTM(lstm_cells, input_shape = (sequence_size,input_nodes)))
    lstm_model.add(layers.Dense(1))
    lstm_model.add(layers.Activation('sigmoid'))
    lstm_model.summary()

    #compile and run the model

    callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="auto")
    lstm_model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"])

    training = lstm_model.fit(
        Training_data_processed,
        Training_labels_processed,
        validation_split=0.1,
        batch_size=100,
        epochs=200,
        callbacks= [callback]
        )

    #save model
    lstm_model.save(outpath + name)
    validation_loss = training.history["loss"]
    np.savetxt(outpath + "val_loss_discriminator" + ".csv",
               validation_loss, delimiter=";")

    plt.close('all')
    fig = plt.figure(figsize=(5, 5), dpi=100, facecolor="w", edgecolor="k")
    plt.box(on=False)
    plt.title("Validation loss")
    plt.plot(range(0, len(validation_loss)), validation_loss, c="g", lw=0.5)

    return


def outputs(model_directory, directory, data_points=9000, sequence_size=150, continuous_recording='True'):
    '''

    :param model_directory: where the model is saved
    :param directory: where the files are saved (type csv, 4 columns, ;)
    :param data_points: number of datapoints to take from the files, has to be a multiple of sequence_size
    :param sequence_size: has to be the sequence size the discriminator has been trained on
    :param continuous_recording: whether the files are continuous (without NaNs) default True
    :return: list with the mean of the outputs of the model for each file in the directory
    '''

    # load model
    model = keras.models.load_model(model_directory + "discriminator")

    # load data
    file_names = glob.glob(os.path.join(directory, "*.csv"))
    fish_complete_1 = np.zeros((len(file_names), data_points, 2))
    fish_complete_2 = np.zeros((len(file_names), data_points, 2))
    outputs = np.zeros((len(file_names)))

    for i in tqdm(range(len(file_names))):
        trajectory = pd.read_csv(file_names[i], delimiter=';')[0:data_points].T
        fish_complete_1[i] = trajectory[0:2].T
        fish_complete_2[i] = trajectory[2:4].T

        # sequence
        data = sequence(fish_complete_1, fish_complete_2, data_points, sequence_size,
                        continuous_recording=continuous_recording)

        # get prediction
        outputs[i] = np.mean(model.predict(data))

    np.savetxt(outpath + "output of discriminator.csv",outputs, delimiter=";")

    return outputs





