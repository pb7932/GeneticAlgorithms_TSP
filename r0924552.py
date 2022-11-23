import Reporter
from TravelingSalesman import TravelingSalesman
import numpy as np

# Modify the class name to match your student number.
class r0924552:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
    def optimize(self, filename):
		# Read distance matrix from file.		
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        tsp = TravelingSalesman(distanceMatrix)

		# Your code here.
        while tsp.convergenceTest():
            tsp.optimize()

            meanObjective = tsp.meanObjective
            bestObjective = tsp.bestObjective
            bestSolution = tsp.bestSolution

			# Your code here.
            print('mean: {} best: {}'.format(meanObjective, bestObjective))

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

		# Your code here.
        return 0

if __name__ == "__main__":
    filename = "tours/tour50.csv"
    alg = r0924552()
    alg.optimize(filename)