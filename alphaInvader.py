import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
# import matplotlib.pyplot as plt


# Definition of Network for Alpha Invader
# Holds everything related to model (definition, history, in charge of loading from memory if necessary)
class Alpha_Invader:
    def __init__(self, input_dim, output_dim):
        # TODO Need to design network architecture

        # Architecture - [CONV - RELU]*2->POOL->[FC->RELU]*2->FC
            # Need 2 matrices of CONV weights, 3 matrices of FC weights
        # Output - Move, Prediction of P(die), Prediction of P(get reward)
            # 1 Value Networ, 2 Policy Networks
        model = models.Sequential()
        model.add(layers.Conv2D(5, (3,3), activation='relu', input_shape=input_dim))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(10, (3,3), activation='relu'))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Flatten()) # Convert out output volume to 1D vector
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(4)) # Output is PMove, PShoot, PWillDieSoon, PWillScoreSoon

        '''
            TODO 
            Look at Pool API
            Look at compile API
            Look at TF for RL
            Look into what kind of optimizer we would want
            Lookup what sess.run does
        '''
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.mean_squared_error,
                      metrics=['accuracy'])
        # model.summary() # Will display structure of model
        pass

# In charge of wrapping network understood params into environment understood params (may want to combine into model)
# Also in charge of pre-processing
class Decision_Process:
    def __init__(self, network):
        self.network = network

    def get_action(self):
        # Use cached network to generate an output
        # TODO call forward_pass on cached network
        # TODO convert forward_pass output into understood action for env
        pass

    def give_feedback(self):
        # Used to pass reward values and other information form environment to cached network
        pass