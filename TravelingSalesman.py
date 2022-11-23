from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import time

class TravelingSalesman:

    def __init__(self, distanceMatrix) -> None:
        self.distanceMatrix = distanceMatrix

        self.lambda_ = 100
        self.mu = 200
        self.K = 3
        self.p = 0.5
        self.mutation_cnt = 20
        self.elitism = 10

        self.meanObjective = 0
        self.meanObjectives = []
        self.bestObjective = 0
        self.bestObjectives = []
        self.bestSolution = []
        self.convergenceCnt = 0

        self.population = self.initializePopulation()
        self.population_fitness = self.evaluation(self.population)

    def optimize(self) -> None:
        """ Main genetic algorithm loop.
        """
        start_time = time.time()

        # create children from selected parents, mutate them and add them to the population
        children = []
        for i in range(self.mu):
            parents = self.selection(self.population, self.population_fitness)

            child = self.recombination(parents)
            child = self.swap_mutation(child)
            children.append(child)


        # eliminate individuals and evaluate them
        self.population += children
        new_fitnesses = self.evaluation(self.population)
        self.population = self.elimination(self.population, new_fitnesses)
        self.population_fitness = self.evaluation(self.population)

        self.get_info()

        elapsed_time = time.time() - start_time
        #print('elapsed time: ',elapsed_time)


    def evaluation(self, pop: list) -> np.ndarray:
        """ Evaluates a given population.
        
        PARAMETERS
        ----------
        pop: population, array of individuals

        RETURNS
        -------
        ndarray: array of fitnesses for the given population
        """
        
        scores = []

        for i in range(len(pop)):
            score = self.getFitnessOfIndividual(pop[i])
            scores.append(score)
        
        return np.array(scores)

    def getFitnessOfIndividual(self, ind: np.ndarray) -> float:
        """ Calculates and returns a fitness of the given inidivdual. 
            Fitness is the sum of distances between cities in the path represented by the inidividual.
        """

        fitness = 0

        for i in range(len(ind) - 2):
            #print(ind)
            dist = self.distanceMatrix[ind[i]][ind[i+1]]

            if dist == np.Inf:
                fitness += 100000
            else:
                fitness += dist
        
        dist = self.distanceMatrix[ind[-1]][ind[0]]
        if dist == np.Inf:
            fitness += 10000
        else:
            fitness += dist

        return fitness
   
    def initializePopulation(self) -> list: # TODO add heuristic initialization
        """ Initializes the population to an array of random permutations starting with zero.
        """
        pop = []

        while len(pop) < self.lambda_:
            ind = np.array([0])
            ind = np.append(ind, np.random.permutation(np.arange(1, len(self.distanceMatrix))))
            pop.append(ind)

        return pop

    def selection(self, pop: list, pop_fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Performs K-tournament selection 

        PARAMETERS
        ----------
        pop: population of individuals
        pop_fitness: fitnesses of the given population

        RETURNS
        -------
        tuple(int, int): selected individuals
        """

        parent1 = pop[self.k_tournament(pop, pop_fitness)]
        parent2 = pop[self.k_tournament(pop, pop_fitness)]

        return (parent1, parent2) 

    def k_tournament(self, pop: list, pop_fitness: np.ndarray) -> int:
        """ Performs K-tournament on given population based on given fitness values.
        
        PARAMETERS
        ----------
        pop: population of individuals
        pop_fitness: fitnesses of the given population

        RETURNS
        -------
        int: index of selected individual
        """
        
        k_tournament = np.random.choice(np.arange(0, len(pop)), self.K)

        best_score = np.Inf
        best_idx = -1

        for i in k_tournament:
            if(best_score > pop_fitness[i]):
                best_score = pop_fitness[i]
                best_idx = i

        return best_idx
    
    def recombination(self, parents: Tuple) -> np.ndarray:
        """ Produces a child by performing order crossover on given parents.
        
        PARAMETERS
        ----------
        parents: two individuals which are to be recombined

        RETURNS
        -------
        np.ndarray: individual created by recombining given parents
        """

        # initialize to zeros, first element is also automatically zero which is in accordance with the representation
        child = np.zeros(len(parents[0]), dtype='int')

        # choose two random crossover points
        idx1 = np.random.choice(range(1, len(child)))
        idx2 = np.random.choice(range(1, len(child)))

        while idx1 == idx2:
            idx2 = np.random.choice(range(1, len(child)))

        # idx1 has to be smaller
        if idx1 > idx2:
            tmp = idx1
            idx1 = idx2
            idx2 = tmp

        # copy fragment from first parent into child
        child[idx1:idx2] = parents[0][idx1:idx2].copy()

        # edges already in child
        used = set(child)

        idx_parent = idx2 - 1
        idx_child = idx2

        while len(used) < len(child):
            idx_parent += 1 

            if idx_parent >= len(child):
                idx_parent = 1
            if idx_child >= len(child):
                idx_child = 1

            if parents[1][idx_parent] in used: 
                continue

            child[idx_child] = parents[1][idx_parent].copy()
            idx_child += 1
            used.add(parents[1][idx_parent].copy())

        return child

    def swap_mutation(self, ind: np.ndarray) -> np.ndarray:
        """ Performs swap mutation with probability <p>.

        Two indices are chosen at random and values at those indices are swapped.
        
        PARAMETERS
        ----------
        ind: individual of the population

        RETURN
        ------
        np.ndarray: mutated individual
        """

        new_ind = ind.copy()
        n = int(np.random.random() * self.mutation_cnt + 1)

        for i in range(n):
            if np.random.random() < self.p:
                idx1 = np.random.choice(range(1, len(new_ind)))
                idx2 = np.random.choice(range(1, len(new_ind)))

                while idx1 == idx2:
                    idx2 = np.random.choice(range(1, len(new_ind)))

                tmp = new_ind[idx1]
                new_ind[idx1] = new_ind[idx2]
                new_ind[idx2] = tmp

        return new_ind

    def elimination(self, pop: list, pop_fitnesses: np.ndarray) -> list: # TODO dont allow duplicates?
        """ Eliminates individuals so that the initial population size is retained.
        Individuals to retain are chosen with k-tournament.
        Elitism is performed.
        """

        new_pop = []

        # elitism
        sorted_pop_idx = np.argsort(pop_fitnesses)
        elit_idx = list(sorted_pop_idx[:self.elitism])
        
        for i in elit_idx:
            new_pop.append(pop[i])

        # k-tournament
        cnt = 0
        remainder = self.lambda_ - self.elitism
        while cnt < remainder:
            ind_idx = self.k_tournament(pop, pop_fitnesses)

            if self.canAdd(ind_idx, elit_idx):
                new_pop.append(pop[ind_idx])
                cnt += 1

        return new_pop
    
    def canAdd(self, ind_idx: int, elit_idx: list) -> bool:
        """ Checks whether an individuals can be added to the population.
        
        An individual can be added to the population if """
        if elit_idx.count(ind_idx) == 0:
            return True
        
        return False

    def convergenceTest(self) -> bool:
        """ Test convergence of this genetic algorithm.

        RETURNS
        -------
        bool: True if for the last (at least) 20 generations the best objective was the same, False otherwise        
        """
        return self.convergenceCnt < 50
    
    def get_info(self):
        """ Gets information about the current population and stores it as member vairables.
        """

        self.meanObjective = np.sum(self.population_fitness) / self.lambda_
        self.meanObjectives.append(self.meanObjective)

        bestIndIdx = np.argmin(self.population_fitness)

        if self.bestObjective == self.population_fitness[bestIndIdx]:
            self.convergenceCnt += 1
        else:
            self.convergenceCnt = 0

        self.bestObjective = self.population_fitness[bestIndIdx]
        self.bestObjectives.append(self.bestObjective)
        self.bestSolution = self.population[bestIndIdx]

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