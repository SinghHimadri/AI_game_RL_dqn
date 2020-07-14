# region IMPORTS
import gym
import numpy as np
# region END


# helper methods for some math
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def relu(vector):
    vector[vector < 0] = 0
    return vector


# for preprocessing images
def downsample(image):
    # Take only alternate pixels halves the resolution of the image
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
    processed_observation = input_observation[35:195]
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1  # everything else (paddles, ball) just set to 1
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


def main():
    # initialize ping pong game (atari style)
    env = gym.make("Pong-v0")

    # get the initial game image
    observation = env.reset()

    # hyperparameters
    # some global variables to fine-tune the learning metrics
    episode_number = 0
    batch_size = 10
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    learning_rate = 1e-4

    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_processed_observations = None

    # weights to determine number of neurons and their relative importance
    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }

    # To be used with rmsprop algorithm (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

    # infinite game loop (need to change) to run neural network and choose an action to take
    while True:
        env.render()

        # generate observations for this and previous frame (trim down data and only give difference between frames)
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        # compute hidden layer and probability of going up by running neural net
        hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)

        # collect observations across the 'episode'
        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        # choose whether to move up or down by 'flipping a coin' given the determined probability
        action = choose_action(up_probability)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)
