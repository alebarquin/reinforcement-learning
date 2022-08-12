import keras
from keras.models import Sequential     # Architecture
from keras.layers import Dense          # Fully-connected layers
from keras.optimizers import Adam       # Backpropagation

# Building the Brain class
class Brain():

     def __init__(self, nInput, nOutput, lr):

          self.nInput = nInput
          self.nOutput = nOutput
          self.LearningRate = lr
     
          # Creating the neural network
          self.model = Sequential()

          self.model.add(Dense(
               units = 32,
               activation = 'relu',
               input_shape = (self.nInput,)
               )
          )

          self.model.add(Dense(
               units = 16,
               activation = 'relu',
               )
          )

          # Not using softmax activation function to the output!
          self.model.add(Dense(
               units = self.nOutput
               )
          )

          self.model.compile(
               optimizer = Adam(learning_rate = self.LearningRate),
               loss = 'mse'
          )