from typing import Tuple
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import time


# TODO initialization
# TODO local search
# TODO recombination?
# TODO mutation
# TODO diversity promotion
# ako je zadnjih 10 istih rjesenja onda napuni populaciju s novim slucajnim ind
class TravelingSalesman:

    def __init__(self, distanceMatrix) -> None:
        self.distanceMatrix = distanceMatrix

        self.lambda_ = 40
        self.mu = 2 * self.lambda_
        self.heurCnt = int(0.2 * self.lambda_) # number of individuals to initialize heuristically
        self.popCnt = 3 # number of populations in the island model
        self.mutation_types = [2,4,5,3,5]
        self.K = 4 # k in k-tournament
        self.p = 0.3 # probability with which to mutate an individual
        if len(self.distanceMatrix < 500):
            self.swap_mutation_cnt = 5 # maximum number of times to do swap mutation on an individual 
            self.inversion_mutation_cnt = 3 # maximum number of times to do inversion mutation on an individual 
        else:
            self.swap_mutation_cnt = 15 # maximum number of times to do swap mutation on an individual 
            self.inversion_mutation_cnt = 8 # maximum number of times to do inversion mutation on an individual 
        self.insert_mutation_cnt = 3 # maximum number of times to do insert mutation on an individual
        self.scramble_mutation_cnt = 2 # maximum number of times to do scramble mutation on an individual
        self.elitism = 2

        self.cnt = 0
        self.population_mix_cnt = int(0.05 * self.lambda_)

        self.meanObjective = 0
        self.meanObjectives = []
        self.stdObjectives = []
        self.bestObjective = 0
        self.bestObjectives = []
        self.bestSolution = []
        self.convergenceCnt = 0


        self.population = []
        self.population_fitness = []

        for i in range(self.popCnt):
            pop = self.initializePopulationHeuristic2(self.lambda_)
            self.population.append(pop)
            self.population_fitness.append(self.evaluation(pop))


    def optimize(self) -> None:
        """ Island model for a genetic algorithm with several different populations.
        """

        self.cnt = self.cnt + 1

        meanObjs = []
        bestObj = np.Infinity
        bestSol = []

        for i in range(self.popCnt):
            (pop, pop_fitness) = self.optimize_pop(self.population[i], self.population_fitness[i], self.mutation_types[i])
            self.population[i] = pop
            self.population_fitness[i] = pop_fitness

            (bestIndIdx, meanObjective) = self.get_info(pop, pop_fitness)

            currBestObj = pop_fitness[bestIndIdx]
            
            if currBestObj < bestObj:
                bestObj = currBestObj
                bestSol = pop[bestIndIdx]

            meanObjs.append(meanObjective)

        # keep track of convergence
        if self.bestObjective == bestObj:
            self.convergenceCnt += 1
        else:
            self.convergenceCnt = 0

        # update stats
        self.meanObjective = np.mean(meanObjs)
        self.meanObjectives.append(meanObjective)
        self.bestObjective = bestObj
        self.bestObjectives.append(bestObj)
        self.bestSolution = bestSol
        self.stdObjectives.append(np.std(meanObjs))

        if self.cnt % 10 == 0:
            self.mix_populations()


    def optimize_pop(self, pop: list, pop_fitness: np.ndarray, mutation_type: int) -> None:
        """ Main genetic algorithm loop for a single population

        PARAMETERS
        ----------
        pop(list): population to perform the genetic algorithm on
        pop_fitness(np.ndarray): array of fitnesses of the given population
        mutation_type(int): represents a mutation method to use in the course of the genetic algorithm
                            1 - insert mutation
                            2 - inverse mutation
                            3 - scramble mutation
                            4 - swap mutation
                            5 - each of the above with equal probability


        RETURNS
        -------
        tuple: population and population fitness, respectively, after performing one loop of the genetic algorithm
        """
        #start_time = time.time()

        # create children from selected parents, mutate them and add them to the population
        children = []

        for i in range(self.mu):
            parents = self.selection(pop, pop_fitness)

            child = self.order_recombination(parents)

            if np.random.random() < self.p:
                match mutation_type:
                    case 1:
                        child = self.insert_mutation(child, self.insert_mutation_cnt)
                    case 2:
                        child = self.inversion_mutation(child, self.inversion_mutation_cnt)
                    case 3: 
                        child = self.scramble_mutation(child)
                    case 4:
                        child = self.swap_mutation(child, self.swap_mutation_cnt)
               
            children.append(child)


        # eliminate individuals and evaluate them
        pop += children
        
        new_fitness = self.evaluation(pop)

        if len(self.distanceMatrix < 250):
            new_fitness = self.fitnessWrapper(pop,new_fitness)

        pop = self.elimination(pop, new_fitness)

        # local search operator
        new_pop = []
        for ind in pop:
            ind = self.lso(ind)
            ind_fitness = self.objectiveValue(ind)
            new_pop.append(ind)

        pop = new_pop
        
        pop_fitness = self.evaluation(pop)

        #self.get_info(pop, pop_fitness)

        #elapsed_time = time.time() - start_time
        #print('elapsed time: ',elapsed_time)

        return (pop, pop_fitness)

    def mix_populations(self):
        """ Exchanges individuals between populations in a random manner. """
        # uzmi random ind iz dvije populacije i pomjesaj ih, moras zamjenit i pop_fitness
        # uzmi sve parove svih populacija

        for i in range(len(self.population)):
            for j in range(len(self.population)):
                if i >= j: continue

                indices = np.random.permutation(range(1, self.lambda_))

                for k in range(self.population_mix_cnt):
                    idx = indices[k]
                    
                    # get random individuals
                    ind1 = self.population[i][idx]
                    ind2 = self.population[j][idx]

                    # recombine them
                    child1 = self.order_recombination((ind1,ind2))
                    child2 = self.order_recombination((ind2,ind1))
                    
                    # add the children into populations
                    self.population[i][idx] = child1
                    self.population[j][idx] = child2


                    # compute their fitnesses
                    self.population_fitness[i][idx] = self.objectiveValue(child1)
                    self.population_fitness[j][idx] = self.objectiveValue(child2)

    def lso(self, ind: np.ndarray) -> np.ndarray:
        """ Uses inversion mutation as a local search operator. (2-opt) - some research papers state this one has the best performance
        """

        best_ind = ind.copy()
        best_ind_fitness = self.objectiveValue(best_ind)

        for i in range(50):
            new_ind = ind.copy()
            new_ind = self.inversion_mutation(new_ind, 1)
            new_ind_fitness = self.objectiveValue(new_ind)

            if (new_ind_fitness < best_ind_fitness):
                best_ind = new_ind
                best_ind_fitness = new_ind_fitness

        return best_ind
            


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
            score = self.objectiveValue(pop[i])
            scores.append(score)
        
        return np.array(scores)

    def objectiveValue(self, ind: np.ndarray) -> float:
        """ Calculates and returns a fitness of the given inidivdual. 
            Fitness is the sum of distances between cities in the path represented by the inidividual.
        """

        fitness = 0

        for i in range(len(ind) - 1):
            dist = self.distanceMatrix[ind[i]][ind[i+1]]

            if dist == np.Inf:
                fitness += 1000000
            else:
                fitness += dist
        
        dist = self.distanceMatrix[ind[-1]][ind[0]]
        if dist == np.Inf:
            fitness += 1000000
        else:
            fitness += dist

        return fitness
   
    def initializePopulation(self, n) -> list: 
        """ Initializes the population to an array of random permutations starting with zero.

        PARAMETERS
        ----------
        n: int
            size of the population
        """
        pop = []

        while len(pop) < n:
            ind = np.array([0])
            ind = np.append(ind, np.random.permutation(np.arange(1, len(self.distanceMatrix))))
            pop.append(ind)

        return pop

    def initializePopulationHeuristic2(self, n:int) -> list:
        pop = []
        ind = [0]
        used = set(ind.copy())
            
        for j in range(len(self.distanceMatrix)-1):
            node = self.nearestNeighbour(ind[-1], used)
            used.add(node)
            ind.append(node)

        # pop.append(np.array(ind))

        for i in range(self.heurCnt+1):
            new_ind = ind.copy()
            new_ind = self.swap_mutation(new_ind,40)
            pop.append(np.asarray(new_ind))

        print(pop)
        pop.extend(self.initializePopulation(n - self.heurCnt))

        return pop

    def initializePopulationHeuristic(self, n: int) -> list:
        """ Initializes the population to an array of random individuals starting with zero. 
        
        A predefined number of those inidividuals are heuristically initialized with the heuristic being the shortest neighbour.

        PARAMETERS
        ----------
        n: int
            size of the population
        """

        pop = []
        nodes = set(range(0, len(self.distanceMatrix)))

        for i in range(self.heurCnt):
            ind = [0]
            ind.append(int(np.random.choice(range(1,len(self.distanceMatrix)))))

           
            used = set(ind.copy())
            
            for j in range(len(self.distanceMatrix)-2):
                if (j % 10 == 0):
                    node = np.random.choice(list(nodes.difference(used)))
                else:
                    node = self.nearestNeighbour(ind[-1], used)

                used.add(node)
                ind.append(node)

            pop.append(np.array(ind))

        print(pop)
        pop.extend(self.initializePopulation(n - self.heurCnt))

        return pop
        
    
    def nearestNeighbour(self, node: int, used: set):
        """ Finds the nearest neighbour of the given node not including the set of used nodes.
        """

        nearestN = 0
        minDist = np.Infinity

        for i in range(len(self.distanceMatrix)):
            if(i not in used and i != node and self.distanceMatrix[node][i] <= minDist):
                nearestN = i
                minDist = self.distanceMatrix[node][i]

        return nearestN


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
            ind_fitness = pop_fitness[i]
            
            if(best_score >ind_fitness):
                best_score = ind_fitness
                best_idx = i

        return best_idx

    def getEdgesOfIndividual(self, ind: np.ndarray) -> list:
        edges = []
        for i in range(len(ind)-1):
            edges.append((ind[i],ind[i+1]))

    def fitnessWrapper(self, pop, pop_fitness):
        popEdges = []

        for ind in pop:
            popEdges.append(self.getEdgesOfIndividual(ind))

        alpha = 0.5
        sigma = len(self.distanceMatrix) / 3

        mod_fitness = []
        for i in range(len(pop)):
            onePlusBeta = 0

            for j in range(len(pop)):
                dist = self.distance(i, j, pop, popEdges)

                if dist <= sigma:
                    onePlusBeta += 1 - (dist/sigma)**alpha
                
            mod_fitness.append(pop_fitness[i] * onePlusBeta)

        return mod_fitness

    def distance(self, idx1: int, idx2: int, pop, popEdges):
        """ Number of not common edges between two individuals. """
        return len(np.setdiff1d(popEdges[idx1], popEdges[idx2]))

    def order_recombination(self, parents: Tuple) -> np.ndarray:
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

    def edge_recombination(self, parents:Tuple):
        """ Produces a child by performing edge crossover on given parents.
        
        PARAMETERS
        ----------
        parents: two individuals which are to be recombined

        RETURNS
        -------
        np.ndarray: individual created by recombining given parents
        """
      
        parent1, parent2 = parents[0], parents[1]#pop[pair[0]], pop[pair[1]]
        child = []
        adj_li = [[] for _ in parent1]
        for idx in range(len(parent1)):
            after_idx = idx + 1 if idx + 1 < len(parent1) else 0
            adj_li[parent1[idx]] += [parent1[idx - 1], parent1[after_idx]]
            adj_li[parent2[idx]] += [parent2[idx - 1], parent2[after_idx]]

        next_elem = 0

        for _ in parent1:
            child.append(next_elem)
            if len(child) == len(parent1):
                break
            self.rm_from_adj_li(adj_li, next_elem)
            next_elem = self.pick_next_elem(adj_li, child[-1])
            if next_elem is None:
                next_elem = np.random.choice(np.setdiff1d(parent1, child))

        return np.array(child)


    def pick_next_elem(self, adj_li, current):
        # Pick doubles
        if len(adj_li[current]) == 0:
            return None
        if len(adj_li[current]) != len(set(adj_li[current])):
            edges = adj_li[current]
            dups = [n for n in edges if edges.count(n) > 1]
            return dups[0]
        lens = [len(adj_li[elem]) for elem in adj_li[current]]
        min = np.argmin(lens)
        if len(lens) != len(set(lens)):
            choices = [n for n in adj_li[current] if len(adj_li[n]) == lens[min]]
            return np.random.choice(choices)
        return adj_li[current][min]


    def rm_from_adj_li(self, adj_li, item):
        for li in adj_li:
            keep_removing = True
            while keep_removing:
                keep_removing = False
                for _ in li:
                    try:
                        li.remove(item)
                        keep_removing= True
                    except ValueError:
                        pass

    def swap_mutation(self, ind: np.ndarray, mutation_cnt: int) -> np.ndarray:
        """ Performs swap mutation with probability <p>.

        Two indices are chosen at random and values at those indices are swapped.
        
        PARAMETERS
        ----------
        ind: individual of the population
        mutation_cnt: maximum number of times to do the swap mutation

        RETURN
        ------
        np.ndarray: mutated individual
        """

        n = int(np.random.random() * mutation_cnt + 1)

        for i in range(n):
            idx1 = np.random.choice(range(1, len(ind)))
            idx2 = np.random.choice(range(1, len(ind)))

            while idx1 == idx2:
                idx2 = np.random.choice(range(1, len(ind)))

            tmp = ind[idx1]
            ind[idx1] = ind[idx2]
            ind[idx2] = tmp

        return ind

    def scramble_mutation(self, ind: np.ndarray) -> np.ndarray:
        """ Performs scramble mutation on the given individual. 
        """

        for i in range(self.scramble_mutation_cnt):
            idx1 = np.random.choice(range(1, len(ind)))
            idx2 = np.random.choice(range(1, len(ind)))

            while idx1 == idx2:
                    idx2 = np.random.choice(range(1, len(ind)))

            if(idx1 > idx2):
                tmp = idx1
                idx1 = idx2
                idx2 = tmp

            ind[idx1:idx2] = np.random.permutation(ind[idx1:idx2])
             
        return ind

     
    def insert_mutation(self, ind: np.ndarray, mutation_cnt: int) -> np.ndarray:
        """ Performs insert mutation on the given individual. 
        """

        n = int(np.random.random() * mutation_cnt + 1)
        
        for i in range(n):
            idx1 = np.random.choice(range(1, len(ind)))
            idx2 = np.random.choice(range(1, len(ind)))

            while idx1 == idx2:
                    idx2 = np.random.choice(range(1, len(ind)))

            if(idx1 > idx2):
                tmp = ind[idx2 + 1:idx1].copy()
                ind[idx1-1] = ind[idx2]
                ind[idx2:idx1-1] = tmp
                # tmp = idx1
                # idx1 = idx2
                # idx2 = tmp
            else:
                tmp = ind[idx1 + 1:idx2].copy()
                ind[idx1 + 1] = ind[idx2]
                ind[idx1 + 2:idx2 + 1] = tmp
        
        return ind


    def inversion_mutation(self, ind: np.ndarray, mutation_cnt: int) -> np.ndarray:
        """ Performs inversion mutation on the given individual. 
        """

        new_ind = ind.copy()
        n = int(np.random.random() * mutation_cnt + 1)
        
        for i in range(n):
            idx1 = np.random.choice(range(1, len(ind)))
            idx2 = np.random.choice(range(1, len(ind)))

            while idx1 == idx2:
                    idx2 = np.random.choice(range(1, len(ind)))

            if(idx1 > idx2):
                tmp = idx1
                idx1 = idx2
                idx2 = tmp

            ind[idx1:idx2+1] = np.flip(ind[idx1:idx2+1])
        
        return ind

              

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
        return self.convergenceCnt < 30
    
    def get_info(self, pop: list, pop_fitness: np.ndarray):
        """ Gets information about the current population and stores it as member vairables.

        PARAMETERS
        ----------
        pop(list): population to perform the genetic algorithm on
        pop_fitness(np.ndarray): array of fitnesses of the given population

        RETURNS
        -------
        int: index of the best individual determined by the given population fitnesses        
        """

        #self.meanObjective = np.sum(pop_fitness) / self.lambda_
        # self.meanObjectives.append(self.meanObjective)
        meanObjective = np.sum(pop_fitness) / self.lambda_

        bestIndIdx = np.argmin(pop_fitness)

        # if self.bestObjective == pop_fitness[bestIndIdx]:
        #     self.convergenceCnt += 1
        # else:
        #     self.convergenceCnt = 0

 
        # self.bestObjective = pop_fitness[bestIndIdx]
        # self.bestObjectives.append(self.bestObjective)
        # self.bestSolution = pop[bestIndIdx]

        return (bestIndIdx, meanObjective)

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