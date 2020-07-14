import gym
import numpy as np

""" Teaching an AI how to play Ping Pong """
# Article: https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0
# Code Source: https://github.com/dhruvp/atari-pong

""" many helper methods """
# helper methods for some math
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

# for preprocessing images
def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

# covnvert color (RGB is the third dimension in the image)
def remove_color(image):
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

# 'flip a coin' to choose what to do
def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # signifies up in openai gym
        return 2
    else:
         # signifies down in openai gym
        return 3

# helper method to do the following:
# 1. crop the imag down to what we care about
# 2. downsample the image
# 3. remove arbitrary details (color) from the image
# 4. convert the image to a 6400x1 matrix for easier computation
# 5. since we only care about what's changed between frames, only store the difference between this and the previous
def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations


# generate a probability of going up using a few steps:
# 1. take dot product of the weights x observation matrix to determine the unprocessed hidden layer values (200x1 matrix of neurons)
# 2. apply reLU on that hidden layer ('introduces the nonlinearities that makes our network capable of computing nonlinear functions rather than just simple linear ones')
# 3. compute outer layer values by taking dot product of hidden layer x weights ['2']
# 4. apply sigmoid function so the result (probability) is between 0 and 1
def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values