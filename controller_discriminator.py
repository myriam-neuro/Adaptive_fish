'''
@author Myriam Hamon
created 12/05/2023
functions to sequence the data for an rnn, taking into consideration the filetype, the continuity of the data and the partners info
and to shuffle and transform it


'''
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


def load_and_sequence(filenames, data_points, sequence_size, filetype, number_of_fish=1, fish_position=[5, 6]):
    '''

    :param filenames: list of the name of the files to load them
    :param data_points: number of datapoints to take from the files
    :param sequence_size: size of the sequence to cut the trajectories into
    :param filetype: type of the file (csv or xlsx)
    :param number_of_fish: by default 1 but can be two
    :param fish_position: in which colums is the actual fish data
    :return: sequenced data
    '''
    # create the arrays
    fish_complete = np.zeros((len(filenames) * number_of_fish, data_points, 2))

    Training_data = np.empty((int(data_points / sequence_size * number_of_fish * len(filenames)), sequence_size, 2))

    # load the data into the arrays
    if filetype == 'csv':
        for i in range(len(filenames)):
            fish_complete[i] = pd.read_csv(filenames[i], delimiter=';')[0:data_points]
    if filetype == 'xlsx' and number_of_fish == 1:
        for i in range(len(filenames)):
            df = pd.read_excel(io=filenames[i], header=None, sheet_name=0)
            fish_complete[i] = df.loc[:, fish_position[0]:fish_position[1]].values[0:data_points]

    if filetype == 'xlsx' and number_of_fish == 2:

        for i in range(0, len(filenames)):
            df = pd.read_excel(io=filenames[i], header=None, sheet_name=0)
            fish_complete[i] = df.loc[:, fish_position[0]:fish_position[1]].values[0:data_points]
        for j in range(0, len(filenames)):
            df = pd.read_excel(io=filenames[j], header=None, sheet_name=0)
            fish_complete[j + len(filenames)] = df.loc[:, 11:12].values[0:data_points]

    # process the data into sequences
    indice = 0
    for i in range(fish_complete.shape[0]):
        for l in range(0, data_points, sequence_size):
            Training_data[indice][:, :2] = fish_complete[i][l:l + sequence_size]

            indice += 1
    if corners:
        corner_expanded = np.repeat(corner, Training_data.shape[0] / len(filenames), axis=0)
        corner_reshape = corner_expanded.reshape(corner_expanded.shape[0], 1, 10)
        corner_reshape = np.repeat(corner_reshape, 100, axis=1)
        walls_expanded = np.repeat(walls, Training_data.shape[0] / len(filenames), axis=0)
        walls_reshape = walls_expanded.reshape(walls_expanded.shape[0], 1, 8)
        walls_reshape = np.repeat(walls_reshape, 100, axis=1)
        Training_data = np.concatenate((Training_data, corner_reshape, walls_reshape), axis=2)

    return Training_data


def load_and_sequence_partners(filenames, data_points, sequence_size, filetype,
                               filenames_partner=False, continuous_recording=True):
    '''

    :param filenames: list of the name of the files to load them
    :param data_points: number of datapoints to take from the files
    :param sequence_size: size of the sequence to cut the trajectories into
    :param filetype: type of the file (csv or xlsx)
    :param filenames_partner : if the data is not in the samefile(for the simulated data, it should go with filetype = csv
    :param continuous_recording  : if the data is continuous in its tracking
    :return: sequenced data of the two fishes in the two different order (fish1-fish2 and fish2-fish1)
    '''
    # create the arrays
    fish_complete_1 = np.zeros((len(filenames), data_points, 2))
    fish_complete_2 = np.zeros((len(filenames), data_points, 2))

    # load the data into the arrays
    if filetype == 'csv':
        for i in tqdm(range(len(filenames))):
            fish_complete_1[i] = pd.read_csv(filenames[i], delimiter=';')[0:data_points]
            df = pd.read_excel(io=filenames_partner[i], header=None, sheet_name=0)
            fish_complete_2[i] = df.loc[:, 11:12].values[0:data_points]

    if filetype == 'xlsx':

        for i in tqdm(range(0, len(filenames))):
            df = pd.read_excel(io=filenames[i], header=None, sheet_name=0)
            fish_complete_1[i] = df.loc[:, 5:6].values[0:data_points]
            fish_complete_2[i] = df.loc[:, 11:12].values[0:data_points]

    # process the data into sequences
    Training_data = sequence(fish_complete_1,fish_complete_2,data_points,sequence_size,continuous_recording=continuous_recording)

    return Training_data

def sequence(fish_1,fish_2,data_points,sequence_size,continuous_recording=True):
    """

    :param fish_1:
    :param fish_2:
    :param data_points:
    :param sequence_size:
    :param continuous_recording:
    :return:
    """

    if continuous_recording:
        data = np.empty((int(data_points / sequence_size * fish_1.shape[0]), sequence_size, 4))

        indice = 0
        for i in tqdm(range(fish_1.shape[0])):
            for l in range(0, fish_1.shape[1], sequence_size):
                data[indice][:, :2] = fish_1[i][l:l + sequence_size]
                data[indice][:, 2:4] = fish_2[i][l:l + sequence_size]
                indice += 1

    else:
        data = np.empty((1, sequence_size, 4))
        for i in range(fish_1.shape[0]):
            # get the indices of the start of continuous segment of sequence_size for the fish with discontinuous data
            segment_starts = find_continuous_segments(fish_2[i], sequence_size)

            # get the sequences of put them in correct form
            for start_index in segment_starts:
                sequence_fish_1 = np.reshape(fish_1[i][start_index:start_index + sequence_size],
                                             (1, sequence_size, 2))
                sequence_fish_2 = np.reshape(fish_2[i][start_index:start_index + sequence_size],
                                             (1, sequence_size, 2))
                data = np.append(data, np.append(sequence_fish_1, sequence_fish_2, axis=2), axis=0)


        data = data[1:]

    return data



def find_continuous_segments(data, min_length):
    current_segment_size = 0
    start_indices = []
    current_start = 0

    for i in range(len(data)):
        if np.count_nonzero(np.isnan(data[i])) == 0:
            if current_segment_size == 0:
                current_start = i
            if current_segment_size <= min_length :
                current_segment_size +=1
            if current_segment_size >= min_length  :
                start_indices.append(current_start)
                current_segment_size = 0

        else :
            current_segment_size = 0

    return start_indices

def transform_shuffle(data,labels, partner):
    """

    :param data:
    :param partner:
    :return: shuffled and transformed data
    """
    data_transformed = np.empty((data.shape))
    data_centered = np.empty((data.shape))

    for i in range(data.shape[0]):
        #center the sequences:
        horizontal_mean = np.mean(np.concatenate((data[i][:,0],data[i][:,2])))
        vertical_mean = np.mean(np.concatenate((data[i][:,1],data[i][:,3])))
        data_centered[i][:,0] = data[i][:,0] - horizontal_mean
        data_centered[i][:,1] = data[i][:,1] - vertical_mean
        data_centered[i][:,2] = data[i][:,2] - horizontal_mean
        data_centered[i][:,3] = data[i][:,3] - vertical_mean

        rotation_angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array(
            [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
        if partner:
            data_transformed[i][:,:2] = np.array([np.array(rotation_matrix @ j.T).T for j in data_centered[i][:,:2]])
            data_transformed[i][:,2:4] = np.array([np.array(rotation_matrix @ j.T).T for j in data_centered[i][:,2:4]])
        else :
            data_transformed[i] = np.array([np.array(rotation_matrix @ j.T).T for j in data_centered[i]])

    # randomly shuffle the data (keras shuffle only shuffle the training data after the training/test split)
    shuffle = np.random.permutation(data.shape[0])
    data_shuffled = data_transformed[shuffle]
    labels_shuffled = labels[shuffle]



    return data_shuffled,labels_shuffled