# Space discover: Genetic Algorithms

import numpy as np
from environment import Environment

# Creating the bots
class Route():

     # Constructor method
     def __init__(self, dnaLength) -> None:
          self.dnaLength = dnaLength
          self.dna = list()
          self.distance = 0

          # Initializing random DNA
          for i in range(self.dnaLength - 1):
               rnd = np.random.randint(1, self.dnaLength)
               while rnd in self.dna:
                    rnd = np.random.randint(1, self.dnaLength)
               self.dna.append(rnd)
          self.dna.append(0)
     
     # Building the Crossover method
     def mix(self, dna1, dna2):
          
          self.dna = dna1.copy()

          for i in range(self.dnaLength - 1):
               # 50% chances of having a mutation for each gene
               if np.random.rand() <= 0.5:
                    previous = self.dna[i]
                    inx = self.dna.index(dna2[i])
                    self.dna[inx] = previous
                    self.dna[i] = dna2[i]
          
          # Random partial mutations 1
          for i in range(self.dnaLength - 1):
               # Only 10% chances of having a random mutation for each gene
               if np.random.rand() <= 0.1:
                    previous = self.dna[i]
                    rnd = np.random.randint(1, self.dnaLength)
                    inx = self.dna.index(rnd)
                    self.dna[inx] = previous
                    self.dna[i] = rnd

               elif np.random.rand() <= 0.1:
                    rnd = np.random.randint(1, self.dnaLength)
                    prevInx = self.dna.index(rnd)
                    self.dna.insert(i, rnd)

                    if i >= prevInx:
                         self.dna.pop(prevInx)
                    else:
                         self.dna.pop(prevInx + 1)

# Initializing the main code
populationSize = 50 # I could define anything here
mutationRate = 0.1 
nSelected = 5 # Number of bots selected for each iteration

env = Environment()
dnaLength = len(env.planets)
population = list()

# Creating the first population
for i in range(populationSize):
     route = Route(dnaLength)
     population.append(route)

# Starting the main loop
generation = 0
bestDist = np.inf # Initial value, will get better

while True:
     generation += 1

     # Evaluating the population
     for route in population:

          env.reset()

          for i in range(dnaLength):
               action = route.dna[i]
               route.distance += env.step(action, 'none')
     
     # Sorting the population
     sortedPopulation = sorted(population, key = lambda x: x.distance) # Sort the bots from the current population so the best performing bots are on top
     population.clear()

     # Update the best distance found
     if sortedPopulation[0].distance < bestDist:
          bestDist = sortedPopulation[0].distance
     
     # Adding the best performing bots to the new population (optional in Genetic Algorithms)
     for i in range(nSelected):
          best = sortedPopulation[i]
          best.distance = 0 # Reset the bot distance so it's not increased on top of that again (while taking a step)
          population.append(best)

     # Filling in the rest of the population
     left = populationSize - nSelected
     for i in range(left):
          newRoute = Route(dnaLength)
          # The new bot could be a complete new random, or an offspring of a previous bot!
          if np.random.rand() <= mutationRate:
               # If it's random, then just append to population list
               population.append(newRoute)
          else:
               # If it's an offspring, then I need a previous bot and mutate it
               inx1 = np.random.randint(0, nSelected)
               inx2 = np.random.randint(0, nSelected)
               while inx1 == inx2:
                    inx2 = np.random.randint(0, nSelected)
               
               dna1 = sortedPopulation[inx1].dna
               dna2 = sortedPopulation[inx2].dna

               newRoute.mix(dna1, dna2)

               population.append(newRoute)
     
     # Displaying the results
     env.reset()

     for i in range(dnaLength):
          action = sortedPopulation[0].dna[i]
          _ = env.step(action, 'normal')

     if generation % 100 == 0:
          env.reset()

          for i in range(dnaLength):
               action = sortedPopulation[0].dna[i]
               _ = env.step(action, 'beautiful')
     
     print('Generation: ' + str(generation) + 'Shortest distance: {:.2f}'.format(bestDist) + 'light years')

