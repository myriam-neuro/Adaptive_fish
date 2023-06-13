"""
@author :  Myriam Hamon
created 31/05/2023

This code goes through every full length trajectory of every fish and plot the histogram of the labels attributed by the discriminator
"""
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
import controller_discriminator as cont



#Setup parameters
random_seed = 10
sequence_size = 150
data_points_robots = 12000
lstm_cells = 32 # same as paper
data_points= 9000
data_points_other =  2250
partner  = 'True'

path_artificial= '.\HAMON\Fish_movements\Artificial_Fish'
file_names_artificial = glob.glob(os.path.join(path_artificial, "*.csv"))

path_real= '..\HAMON\Fish_movements\Real_Fish'
file_names_real_two_fish = glob.glob(os.path.join(path_real, "*.xlsx"))

path_real_one_fish  = '.\HAMON\Fish_movements\Real_Fish\Robotracker'
file_names_real_one_fish = glob.glob(os.path.join(path_real_one_fish, "*.xlsx"))

path_robot= '.\HAMON\Fish_movements\Artificial_Fish'
file_names_robot = glob.glob(os.path.join(path_robot, "*.xlsx"))

path_artificial_partner = '.\HAMON\Fish_movements\Artificial_Fish\Partners'
file_names_partner = glob.glob(os.path.join(path_artificial_partner, "*.xlsx"))


file_names_partner_1 = np.repeat(file_names_partner[0], 25)
file_names_partner_2 = np.repeat(file_names_partner[1],18)

outpath = './Output_Discriminator/'
import os
isExist = os.path.exists(outpath)
if not isExist:
    os.makedirs(outpath)
    print("Made new output folder!")


model = keras.models.load_model("final_discriminator"+str(sequence_size)+ '_'+str(data_points) +'_'+ str(partner))


for i in range (0,25):
    """ this loop goes through the 25 datafiles generated with N09P4"""
    fish_complete_1 = pd.read_csv(file_names_artificial[i], delimiter=';')[0:data_points_other]
    fish_complete_1 = np.array(fish_complete_1).reshape(1,data_points_other,2)
    df = pd.read_excel(io=file_names_partner_1[i], header=None, sheet_name=0)
    fish_complete_2 = df.loc[:, 11:12].values[0:data_points_other]
    fish_complete_2 = np.array(fish_complete_2).reshape(1,data_points_other,2)
    data = cont.sequence(fish_complete_1,fish_complete_2,data_points_other,sequence_size)
    prediction = model.predict(data)

    fig,ax = plt.subplots(2)
    ax[0].plot(fish_complete_1[0][:,0],fish_complete_1[0][:,1], label ='artificial')
    ax[0].plot(fish_complete_2[0][:,0],fish_complete_2[0][:,1], label ='real')
    ax[0].legend()
    ax[1].hist(prediction)
    ax[1].axvline(np.mean(prediction))
    fig.suptitle('mean output ='+  str(np.mean(prediction)))
    plt.savefig(outpath+'N09P4'+str(i))
    plt.close(fig)


for i in range (0,18):
    """ this loop goes through the 18 datafiles generated with N12P4 (number of datapoints generated is different)"""
    fish_complete_1 = pd.read_csv(file_names_artificial[25+i], delimiter=';')[0:data_points]
    fish_complete_1 = np.array(fish_complete_1).reshape(1,data_points,2)
    df = pd.read_excel(io=file_names_partner_2[i], header=None, sheet_name=0)
    fish_complete_2 = df.loc[:, 11:12].values[0:data_points]
    fish_complete_2 = np.array(fish_complete_2).reshape(1,data_points,2)
    data = cont.sequence(fish_complete_1,fish_complete_2,data_points,sequence_size)
    prediction = model.predict(data)

    fig,ax = plt.subplots(2)
    ax[0].plot(fish_complete_1[0][:,0],fish_complete_1[0][:,1], label ='artificial')
    ax[0].plot(fish_complete_2[0][:,0],fish_complete_2[0][:,1], label ='real')
    ax[0].legend()
    ax[1].hist(prediction)
    ax[1].axvline(np.mean(prediction))
    fig.suptitle('mean output ='+  str(np.mean(prediction)))
    plt.savefig(outpath+'N12P4'+str(i))
    plt.close(fig)