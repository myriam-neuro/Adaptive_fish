# Adaptive_fish
Code generated during lab rotation at the Adaptive Systems lab (HU Berlin) on the project Prediction and Social Interaction in fish

The file discriminator takes all of the training data files and feed them into the LSTM model to train it. Final output of the file : Trained network
The file training discriminator is basically the same as the file discriminator but contains a function that takes into input the directory of the files and trains the network, and save that network
In order for the discriminitor file to work, load the controller_discriminator file that contains all of the functions used in the discriminator

The file working memory creates a lot of different RNNs with different sequences sizes as input and evaluates those network with a trained discriminator in order to evaluate the optimal sequence size length
The file test model takes the training data and feeds it to the trained network and then plot the histograms of the outputs of the discriminator
The file plot outputs looks at how the simulated fish trajectories are labelled by the trained discriminator

