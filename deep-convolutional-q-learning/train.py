# Snake training

from environment import Environment
from brain import Brain
from dqn import DQN
import numpy as np
import matplotlib.pyplot as plt

# Defining the parameters
learningRate = 0.0001
maxMemory = 60000
gamma = 0.9              # Discount factor for the Q values
batchSize = 32
nLastStates = 4          # How many frames I stack on top of each others

epsilon = 1.
epsilonDecayRate = 0.0002     # Exploration
minEpsilon = 0.05

filepathToSave = 'model2.h5'

# Initializing the environment
env = Environment(waitTime=0)
brain = Brain(
     inputShape = (env.nColumns, env.nRows, nLastStates),
     lr = learningRate
     )
model = brain.model
dqn = DQN(
     maxMemory = maxMemory,
     discount = gamma
)

# Building a function that will reset current and next states
def resetStates():

     currentState = np.zeros((1, env.nColumns, env.nRows, nLastStates))

     for i in range(nLastStates):
          currentState[0, :, :, i] = env.screenMap
     
     # When the game resets, current state is the same as next state, so return same for both
     return currentState, currentState

# Starting the main loop
epoch = 0           # How many games I've played
nCollected = 0      # How many apples I've eaten
maxNCollected = 0   # Max apples collected in a single round
totNCollected = 0   # Total amount of apples collected in 100 games
scores = list()     # Everygame append the total apples collected to the list to track performance

while True:

     epoch += 1

     # Reseting the environment and starting to play the game
     env.reset()
     currentState, nextState = resetStates()
     gameOver = False

     while not gameOver:
          
          # Selecting an action to play (Epsilon greedy)
          if np.random.rand() <= epsilon:
               action = np.random.randint(0, 4) # I have 4 possible actions (0 to 3). Numpy randint() method excludes the upper bands
          else:
               qvalues = model.predict(currentState)[0] # Keras returns a 2D array with one row. I just need the first one
               action = np.argmax(qvalues)
          
          # Updating the environment
          frame, reward, gameOver = env.step(action)
          # Transform frame (2D) to match currentState dimensions (4D). The first dimension is always required by Keras
          frame = np.reshape(frame, (1, env.nColumns, env.nRows, 1))
          # Append that last frame to nextState
          nextState = np.append(nextState, frame, axis = 3)
          # Since I've appended a new frame, now nextState is greater than 4, so I need to delete the first one
          nextState = np.delete(nextState, 0, axis = 3)