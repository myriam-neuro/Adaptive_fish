'''

@author Myriam Hamon
created on 03/05/2023
this code takes one fish's trajectory and evaluate its probability of being biological
So far the model is trained on real fish trajectories and poorly simulated fish trajectories and robotic fish trajectories

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
sys.path.append("../Fish_movements")
import geometry_tools as geom
import controller_discriminator as cont

##
#Setup parameters
random_seed = 10
sequence_size = 150
data_points_robots = 12000
lstm_cells = 64
data_points= 9000
data_points_other=2250
corner_size = 9
partner = True


#Get the data files names

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

##

if partner:
    input_nodes = 4
    file_names_partner_1 = np.repeat(file_names_partner[0], 25)
    file_names_partner_2 = np.repeat(file_names_partner[1],18) #this is temporary, once proper simulation are done remove it
    Training_data_real = cont.load_and_sequence_partners(file_names_real_two_fish, data_points, sequence_size, 'xlsx')
    Training_data_robot = cont.load_and_sequence_partners(file_names_robot, data_points_robots, sequence_size, 'xlsx', continuous_recording = False)
    Training_data_artificial_1 = cont.load_and_sequence_partners(file_names_artificial[0:25], data_points_other, sequence_size, 'csv', filenames_partner = file_names_partner_1 )
    Training_data_artificial_2 = cont.load_and_sequence_partners(file_names_artificial[25:25+18], data_points, sequence_size, 'csv', filenames_partner = file_names_partner_2 )

    Training_data_fake = Training_data_robot #np.concatenate((Training_data_artificial_1,Training_data_artificial_2))

else :
    input_nodes = 2
    Training_data_one_fish_real = cont.load_and_sequence(file_names_real_one_fish, data_points,sequence_size,'xlsx', fish_position  = [11,12])
    Training_data_two_fish_real = cont.load_and_sequence(file_names_real_two_fish, data_points,sequence_size,'xlsx',number_of_fish = 2)
    Training_data_artificial = cont.load_and_sequence(file_names_artificial, data_points,sequence_size,'csv')
    Training_data_robot = cont.load_and_sequence(file_names_robot,data_points,sequence_size,'xlsx')
    Training_data_real = np.concatenate((Training_data_two_fish_real, Training_data_two_fish_real))
    Training_data_fake = np.concatenate((Training_data_artificial, Training_data_robot), axis=0)

##

Training_labels_real = np.ones(Training_data_real.shape[0])
Training_labels_fake = np.zeros (Training_data_fake.shape[0])
print(Training_data_real.shape)
print(Training_data_fake.shape)
Training_data = np.concatenate((Training_data_real, Training_data_fake))
Training_labels = np.concatenate((Training_labels_real, Training_labels_fake))


print(Training_labels.shape)

Training_data_processed,Training_labels_processed = cont.transform_shuffle(Training_data, Training_labels, partner)

print(Training_data.shape)
##

#Build model
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
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
lstm_model.save("final_discriminator"+str(sequence_size)+ '_'+str(data_points) +'_'+ str(partner))

plt.plot(training.history['val_loss'])
plt.title('Val_Loss')
plt.show()

