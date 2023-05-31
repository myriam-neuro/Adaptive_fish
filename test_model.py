'''
@author Myriam Hamon
created on 03/05/2023
this code take a model and look at the distribution of its prediction on data
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
import controller_discriminator as cont



#Setup parameters
random_seed = 10
sequence_size = 150
data_points_robots = 12000
lstm_cells = 32 # same as paper
data_points= 9000
data_points_other =  2250
partner  = 'True'

path_artificial= '..\Fish_movements\Artificial_Fish'
file_names_artificial = glob.glob(os.path.join(path_artificial, "*.csv"))

path_real= '..\Fish_movements\Real_Fish'
file_names_real_two_fish = glob.glob(os.path.join(path_real, "*.xlsx"))

path_real_one_fish  = '..\Fish_movements\Real_Fish\Robotracker'
file_names_real_one_fish = glob.glob(os.path.join(path_real_one_fish, "*.xlsx"))

path_robot= '..\Fish_movements\Artificial_Fish'
file_names_robot = glob.glob(os.path.join(path_robot, "*.xlsx"))

path_artificial_partner = '..\Fish_movements\Artificial_Fish\Partners'
file_names_partner = glob.glob(os.path.join(path_artificial_partner, "*.xlsx"))



file_names_partner_1 = np.repeat(file_names_partner[0], 25)
file_names_partner_2 = np.repeat(file_names_partner[1],
                                 18)  # this is temporary, once proper simulation are done remove it
Training_data_fish = cont.load_and_sequence_partners(file_names_real_two_fish, data_points,
                                                                                  sequence_size, 'xlsx')
Training_data_robot = cont.load_and_sequence_partners(file_names_robot,
                                                                                    data_points_robots, sequence_size,
                                                                                    'xlsx', continuous_recording=False)
Training_data_artificial_1 = cont.load_and_sequence_partners(
    file_names_artificial[0:25], data_points_other, sequence_size, 'csv', filenames_partner=file_names_partner_1)
Training_data_artificial_2 = cont.load_and_sequence_partners(
    file_names_artificial[25:25 + 18], data_points, sequence_size, 'csv', filenames_partner=file_names_partner_2)

outpath = './Output_Discriminator/'
import os
isExist = os.path.exists(outpath)
if not isExist:
    os.makedirs(outpath)
    print("Made new output folder!")


model = keras.models.load_model("final_discriminator"+str(sequence_size)+ '_'+str(data_points) +'_'+ str(partner))


#This is to look at all data files
Training_data_real = Training_data_fish
Training_data_artificial = np.concatenate(((Training_data_artificial_2,Training_data_artificial_1)))

prediction_artificial = model.predict(Training_data_artificial)
prediction_robot = model.predict(Training_data_robot)
prediction_real = model.predict(Training_data_real)
plt.hist(prediction_artificial, label = 'artificial', color = 'blue', histtype= 'step',bins=20)
plt.hist(prediction_real, label= 'real', color = 'red', histtype='step', bins = 20 )
plt.hist(prediction_robot, label='robot',color='green', histtype= 'step',bins=20)
plt.axvline(np.mean(prediction_robot), color = 'green')
plt.axvline(np.mean(prediction_real), color = 'red')
plt.axvline(np.mean(prediction_artificial), color = 'blue')
plt.title("Distribution of prediction of the final discriminator")
plt.xlabel('prediction')
plt.legend()
plt.savefig("Distribution of ouputs final discriminator")
plt.show()

