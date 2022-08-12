# Snake: Deep Convolutional Q-Learning

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# Building the Brain class
class Brain():

     def __init__(self, inputShape, lr = 0.005):
          self.inputShape = inputShape
          self.learningRate = lr
          self.numOutputs = 4

          # Creating the neural network
          self.model = Sequential()

          self.model.add(Conv2D(
               filters=32,
               kernel_size=(3,3),
               activation='relu',
               input_shape=self.inputShape
          ))

          self.model.add(MaxPooling2D(
               pool_size=(2, 2) # This means the image will be x2 smaller
          ))

          self.model.add(Conv2D(
               64,
               (2, 2),
               activation='relu'
          ))

          self.model.add(Flatten())

          self.model.add(Dense(
               units=256,
               activation='relu'
          ))

          self.model.add(Dense(
               units=self.numOutputs
               # No activation function, means I'm going with Linear. Could try Softmax instead
          ))

          self.model.compile(
               optimizer=Adam(learning_rate=self.learningRate),
               loss='mse'
          )
     
     # Building a method that will load a model
     def loadModel(self, filepath):
          self.model = load_model(filepath)
          return self.model