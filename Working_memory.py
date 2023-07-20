'''
@author Myriam Hamon
created on 03/05/2023
this code take a model and look at the distribution of its prediction on the training data
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
lstm_cells = 32 # same as paper
data_points= 8850
input_nodes = 4


path_artificial_ftrack= r'C:\\Users\HAMON\Fish_movements\Working_memory\follow track'
file_names_ftrack = glob.glob(os.path.join(path_artificial_ftrack, "*.csv"))


path_artificial_fpartner= r'C:\\Users\HAMON\Fish_movements\Working_memory\follow partner'
file_names_fpartner = glob.glob(os.path.join(path_artificial_fpartner, "*.csv"))

path_artificial_partner = r'C:\\Users\HAMON\Fish_movements\Artificial_Fish\Partners'
file_names_partner = glob.glob(os.path.join(path_artificial_partner, "*.xlsx"))






outpath = './Working_memory_outputs/'
import os
isExist = os.path.exists(outpath)
if not isExist:
    os.makedirs(outpath)
    print("Made new output folder!")


model = keras.models.load_model("final_discriminator_trained")

mean_fpartner = []
mean_ftrack = []
var_fpartner = []
var_ftrack = []
#This is to look at all data files
for i in range(len(file_names_fpartner)):
    fish_complete_1_fpartner = pd.read_csv(file_names_fpartner[i], delimiter=';')[0:data_points]
    fish_complete_1_fpartner = np.array(fish_complete_1_fpartner).reshape(1,data_points,2)
    df = pd.read_excel(io=file_names_partner[1], header=None, sheet_name=0)
    fish_complete_2_fpartner = df.loc[:, 11:12].values[0:data_points]
    fish_complete_2_fpartner = np.array(fish_complete_2_fpartner).reshape(1,data_points,2)
    data = cont.sequence(fish_complete_1_fpartner,fish_complete_2_fpartner,data_points,sequence_size)
    prediction = model.predict(data)
    mean_fpartner.append(np.mean(prediction))
    var_fpartner.append(np.var(prediction))

    fish_complete_1_ftrack = pd.read_csv(file_names_ftrack[i], delimiter=';')[0:data_points]
    fish_complete_1_ftrack = np.array(fish_complete_1_ftrack).reshape(1, data_points, 2)
    df = pd.read_excel(io=file_names_partner[1], header=None, sheet_name=0)
    fish_complete_2_ftrack = df.loc[:, 11:12].values[0:data_points]
    fish_complete_2_ftrack = np.array(fish_complete_2_ftrack).reshape(1, data_points, 2)
    data_2 = cont.sequence(fish_complete_1_ftrack, fish_complete_2_ftrack, data_points, sequence_size)
    prediction_2 = model.predict(data_2)
    mean_ftrack.append(np.mean(prediction_2))
    var_ftrack.append(np.var(prediction_2))

plt.errorbar(range(2,25,2),mean_fpartner[0:12],yerr = var_fpartner[0:12], label = 'Follow partner, 1 layer')
plt.errorbar(range(2,25,2), mean_fpartner[12:] ,yerr = var_fpartner[12:], label = 'Follow partner, 2 layers')
plt.errorbar(range(2,25,2),mean_ftrack[0:12],yerr = var_ftrack[0:12], label = 'Follow track, 1 layer')
plt.errorbar(range(2,25,2), mean_ftrack[12:],yerr = var_fpartner[12:], label = 'Follow track, 2 layers')
plt.legend()
plt.title('Realness of the interaction based on the working memory')
plt.show()



