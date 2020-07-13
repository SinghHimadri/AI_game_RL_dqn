# region IMPORTS
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# region END


# setting up environment
# the game we are training for
ENV_NAME = "CartPole-v1"

# hyperparameters

GAMMA = 0.95  # decay rate to calculate the future discounted reward
LEARNING_RATE = 0.001  # neural net learns in each iteration

MEMORY_SIZE = 1000000  # limits on RAM size to ensure this runs smoothly
BATCH_SIZE = 20  # number of training examples used in oen iteration

EXPLORATION_MAX = 1.0  # agent randomly decides its action not predicting
EXPLORATION_MIN = 0.01  # agent to explores at least this amount
EXPLORATION_DECAY = 0.995  # decrease explorations as model improves


# implementation of Deep Q Network algorithm (reinfocement learning)

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

    # a list of previous experiences and observations to re-train the model with the previous experiences
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # agent will randomly select its action at first by a certain percentage ‘exploration rate’
    # then agent will predict the reward value based on the current state and pick the action that will give the highest reward

    def act(self, state):

        if np.random.rand() < self.exploration_rate:
            # The agent acts randomly
            return random.randrange(self.action_space)

        # Predict the reward value based on the given state
        q_values = self.model.predict(state)

        # Pick the action based on the predicted reward
        return np.argmax(q_values[0])

    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return
        # Sample batch from the memory
        batch = random.sample(self.memory, BATCH_SIZE)
        # Extract informations from each memory
        for state, action, reward, state_next, dead in batch:
            # if done, make our target reward
            q_update = reward
            if not dead:
                # predict the future discounted reward
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            # make the agent to approximately map
            # the current state to future discounted reward
            # We'll call that q_values
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole():
    # initialize game and score logger
    env = gym.make(ENV_NAME)
    # tool created to display 'score'
    # score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0

    # continue indefinitely to train and improve the mdoel
    while True:
        run += 1
        # a new frame
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0

        # keep stepping till the cart falls out of frame
        while True:

            # more steps = more successful (even if it is moving)
            step += 1
            env.render()

            # choose action
            action = dqn_solver.act(state)

            # analyze
            state_next, reward, dead, info = env.step(action)

            # reinforce positive outcome , penalize bad outcome
            reward = reward if not dead else -reward
            state_next = np.reshape(state_next, [1, observation_space])

            # memorize this iteration to shape the following
            dqn_solver.memorize(state, action, reward, state_next, dead)
            state = state_next

            if dead:

                # score = # of steps taken in a particular run (too many steps is bad)
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                # score_logger.add_score(step, run)
                break
            dqn_solver.replay()


cartpole()
