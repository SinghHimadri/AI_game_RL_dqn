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


def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195]
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)  # convert to black and white image
    processed_observation = remove_background(processed_observation)  # convert image to 6400x1 matrix easier computation
    processed_observation[processed_observation != 0] = 1  # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()
    # subtract the previous frame from the current one so we are only processing on changes in the game
    # we only care about what's changed between frames
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations


# generate a probability of going up using a few steps:
# 1. take dot product of the weights.observation matrix to determine the unprocessed hidden layer values (200x1 matrix of neurons)
# 2. apply reLU on that hidden layer ('introduces the nonlinearities that makes our network capable of computing nonlinear functions rather than just simple linear ones')
# 3. compute outer layer values by taking dot product of hidden layer(200X1) x weights ['2'](1X200)
# 4. apply sigmoid function so the result (probability) is between 0 and 1
def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values


'''
Our goal is to find ∂C/∂w1 (BP4), the derivative of the cost function with
respect to the first layer’s weights, and ∂C/∂w2, the derivative of the
cost function with respect to the second layer’s weights. These gradients
will help us understand what direction to move our weights in for the greatest improvement.
'''


# compute gradient for backpropogation
def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }


# apply RMSProp, an algorithm for updating weights
def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name])  # reset batch gradient buffer


# actions taken later in the game carry more weight
def discount_rewards(rewards, gamma):
    """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
    This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    """ discount the gradient with the normalized rewards """
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards


def main():
    # initialize ping pong game (atari style)
    env = gym.make("Pong-v0")

    # get the initial game image
    observation = env.reset()

    # hyperparameters
    episode_number = 0
    batch_size = 10  # how many episodes to wait before moving the weights
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    learning_rate = 1e-4

    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_processed_observations = None

    # weights to determine number of neurons and their relative importance
    # initialize each layer’s weights with random numbers and normalize
    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }

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

        # the 'learning' stage

        # treat the action as the 'right' move to determine the derivative
        # the derivative reflects the answer to: 'How does changing the output probability (of going up) affect my result of winning the round?'
        # see here: http://cs231n.github.io/neural-networks-2/#losses
        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        # now we have a gradient per action
        episode_gradient_log_ps.append(loss_function_gradient)

        # compute the 'policy gradient' to determine 'how' we learn
        # if we won, keep doing the same thing, otherwise, generate less of such actions that made us lose

        if done: # an episode finished
            episode_number += 1
            # once the round is done, compile all the observations and gradients
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            # Tweak the gradient of the log_ps based on the discounted rewards
            # this puts less weight on actions taken at the beginning of the game vs the end
            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)
            gradient = compute_gradient(
                episode_gradient_log_ps_discounted,
                episode_hidden_layer_values,
                episode_observations,
                weights
            )

            # Sum the gradient for use when we hit the batch size
            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            if episode_number % batch_size == 0:
                update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] # reset values
            observation = env.reset() # reset env
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print (f'resetting env. episode reward total was {reward_sum}. running mean: {running_reward}')
            print(episode_number)
            reward_sum = 0
            prev_processed_observations = None

main()
