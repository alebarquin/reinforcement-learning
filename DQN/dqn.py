import numpy as np

# Initializing the Experience Replay memory
class DQN():

     def __init__(self, maxMemory, discount):
          self.maxMemory = maxMemory
          self.discount = discount
          self.memory = list()

     # Remembering new experience
     def remember(self, transition, gameOver):
          self.memory.append([transition, gameOver])
          # Check if max memory was reached. If so, delete the oldest experience
          if len(self.memory) > self.maxMemory:
               del self.memory[0]

     # Getting batches of inputs and targets
     def getBatch(self, model, batchSize):
          lenMemory = len(self.memory)
          nInput = self.memory[0][0][0].shape[1] # First experience, transition, currentState. Shape[1] for the number of columns equals the number of inputs
          nOutput = model.output_shape[-1] # The number of layers from the last (output) layer

          # Initializing the inputs and targets
          input = np.zeros((min(batchSize, lenMemory), nInput)) # I cannot hvae a batch bigger than my actual memory
          target = np.zeros((min(batchSize, lenMemory), nOutput))

          # Extracting transitions from random experiences
          for i, inx in enumerate(np.random.randint(0, lenMemory, size = min(batchSize, lenMemory))):
               currentState, action, reward, nextState = self.memory[inx][0]
               gameOver = self.memory[inx][1]

               # Updating input and target
               input[i] = currentState
               target[i] = model.predict(currentState)[0]
               if gameOver:
                    target[i][action] = reward
               else:
                    target[i][action] = reward + self.discount * np.max(model.predict(nextState)[0])
               
          return input, target