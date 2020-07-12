import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# the game we are training for
ENV_NAME = "CartPole-v1"

# hyperparameters to determine leraning rate and stuff
GAMMA = 0.95 # how far the influence of a single training example reaches
LEARNING_RATE = 0.001 # how quickly model adapts to the problem

# limits on RAM size to ensure this runs smoothly
MEMORY_SIZE = 1000000
BATCH_SIZE = 20 # number of training examples used in oen iteration

# helps the agent's actions converge from being random to become more consistent
# becomes more conservative over time

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

""" implementation of deep q learning algorithm (reinfocement learning) """
class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        # initialize a model
        self.model = Sequential()

        # add three layers to the model: relu, relu, and linear
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))

        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    # memorize each 'frame' or iteration of the 'game'
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # either do something random or act according to the prediction of the model
    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])


    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, dead in batch:
            q_update = reward
            if not dead:
                # reward/punish behavior accordingly
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
