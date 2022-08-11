from brain import Brain
from dqn import DQN

import gym
import numpy as np
import matplotlib.pyplot as plt

# Setting the parameters
learningRate = 0.001
maxMemory = 5000
gamma = 0.9
batchSize = 32

# Adaptative Epsilon greedy model!
epsilon = 1.
epsilonDecayRate = 0.995

# Initializing the Environment, the Brain and the Experience Replay Memory
env = gym.make('MountainCar-v0')
brain = Brain(
     nInput = 2, # I have position and velocity as input arguments
     nOutput = 3, # Left, right or staying still
     lr = learningRate
     )
model = brain.model
dqn = DQN(
     maxMemory = maxMemory,
     discount = gamma
     )

# Starting the main loop
epoch = 0
currentState = np.zeros((1, 2)) # Keras requires a 2D array. In this case I have two inputs, position and velocity
nextState = currentState
totalReward = 0
rewards = list()

while True:

     # Every time I enter this loop, it's a new game, so increment epoch
     epoch += 1

     # Starting to play the game
     env.reset()
     currentState = np.zeros((1, 2))
     nextState = currentState
     gameOver = False

     while not gameOver:

          # Taking an action (Epsilon greedy)
          if np.random.rand() <= epsilon:
               action = np.random.randint(0, 3) # 0: Left, 1: Staying, 2: Right
          else:
               qvalues = model.predict(currentState)[0] # model.predict returns a 2D array, but I'm only interested in the rows so I get the Q values
               action = np.argmax(qvalues)
          
          # Updating the Environment
          nextState[0], reward, gameOver, _ = env.step(action)
          env.render()

          totalReward += reward

          # Remembering new experience, training the AI and updating the current state
          dqn.remember([currentState, action, reward, nextState], gameOver)
          input, target = dqn.getBatch(model, batchSize)
          model.train_on_batch(input, target)

          currentState = nextState
     
     # Lowering the epsilon and displaying the results
     epsilon *= epsilonDecayRate

     print('Epoch: ' + str(epoch) + ' Epsilon: {:.5f}'.format(epsilon) + ' Total reward: {:.2f}'.format(totalReward))

     rewards.append(totalReward)
     totalReward = 0
     plt.plot(rewards)
     plt.xlabel('Epoch')
     plt.ylabel('Rewards')
     plt.show()

env.close()
