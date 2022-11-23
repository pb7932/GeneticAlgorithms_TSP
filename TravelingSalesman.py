import numpy as np
import matplotlib.pyplot as plt
import time

class TravelingSalesman:

    def __init__(self, distanceMatrix) -> None:
        self.distanceMatrix = distanceMatrix

        self.lambda_ = 200
        self.mu = 400
        self.K = 3
        self.p = 0.5
        self.elitism = 50

        self.meanObjective = 0
        self.meanObjectives = np.array([])
        self.bestObjective = 0
        self.bestObjectives = np.array([])
        self.bestSolution = np.array([])

        self.population = self.initializePopulation()
        self.population_fitness = np.array([])
        self.evaluate(self.population)

    def optimize(self) -> None:
        """ Main genetic algorithm loop.
        """
        start_time = time.time()

        # create children from selected parents, mutate them and add them to the population
        for i in range(self.mu):
            parents = self.selection(self.population)
            child = self.recombination(parents)
            child = self.mutation(child)
            
            self.population.put(child)

        # eliminate individuals and evaluate them
        new_fitnesses = self.evaluation(self.population)
        self.elimination(self.population, new_fitnesses)
        self.population_fitness = self.evaluation(self.population)

        self.get_info()

        elapsed_time = time.time() - start_time
        print('elapsed time: ',elapsed_time)


    def evaluation(self, population: np.ndarray) -> np.ndarray:
        """ Evaluates a given population.
        
        PARAMETERS
        ----------
        population: array of individuals

        RETURNS
        -------
        ndarray: array of fitnesses for the given population
        """
        
        scores = np.array([])

        for i in range(len(population)):
            score = self.getFitnessOfIndividual(population[i])
            scores.put(score)
        
        return scores

    def getFitnessOfIndividual(self, individual: np.ndarray) -> float:
        """ Calculates and returns a fitness of the given inidivdual. 
            Fitness is the sum of distances between cities in the path represented by the inidividual.
        """

        fitness = 0

        for i in range(len(individual) - 1):
            dist = self.distanceMatrix[individual[i]][individual[i+1]]

            if dist == np.Inf:
                fitness += 10000
            else:
                fitness += dist
        
        dist = self.distanceMatrix[individual[-1]][individual[0]]
        if dist == np.Inf:
            fitness += 10000
        else:
            fitness += dist

        return fitness
   
    def initializePopulation(self) -> np.ndarray:
        """ Initializes the population to an array of random permutations starting with zero.
        """
        pop = np.array([])

        while len(pop) < self.lambda_:
            ind = np.array([0])
            ind = np.append(ind, np.random.permutation(np.arange(1, len(self.distanceMatrix))))

        return pop

    def selection(self, population: np.ndarray):
        pass

    def recombination(self):
        pass

    def mutation(self, individual: np.ndarray):
        pass

    def elimination(self, population: np.ndarray):
        pass

    def convergenceTest(self) -> bool:
        """ Test convergence of this genetic algorithm.

        
        """
        return False
    
    def get_info(self):
        """
        Gets information about the current population and stores it as member vairables
        """
        # scores = [s for s in self.population_score]# if s != np.Inf]
        # self.meanObjective = np.sum(np.array(scores)) / len(scores)
        # self.meanObjectives.append(self.meanObjective)

        # bestObjIdx = np.argmin(self.population_score)

        # # for convergence test
        # if self.bestObjective == self.population_score[bestObjIdx]:
        #     self.bestSameCnt += 1
        # else:
        #     self.bestSameCnt = 0

        # self.bestObjective = self.population_score[bestObjIdx]
        # self.bestSolution = self.population[bestObjIdx]
        # self.bestObjectives.append(self.bestObjective)
        pass

    def plotObjective(self):
        """ Plots the mean and best with respect to time.
        """

        plt.plot(self.meanObjectives, label='mean')
        plt.plot(self.bestObjectives, label='best')

        plt.title('Mean and best objectives')
        plt.xlabel('time')
        plt.ylabel('objective')

        plt.legend()
        plt.show()