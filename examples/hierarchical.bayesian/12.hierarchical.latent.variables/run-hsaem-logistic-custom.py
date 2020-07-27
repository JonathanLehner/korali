import sys
sys.path.append('./_model/logistic')
sys.path.append('./_model')
from model import *
from utils import generate_variable

import numpy as np
import korali


def main():
  # Initialize the distribution
  distrib = LogisticConditionalDistribution()

  k = korali.Engine()
  e = korali.Experiment()

  data_vector = [[] for _ in range(distrib._p.nIndividuals)]
  for i in range(distrib._p.nIndividuals):
    data_vector[i] = distrib._p.data[i].tolist()

  e["Problem"]["Type"] = "Bayesian/Latent/HierarchicalLatentCustom"


  # The computational models for the log-likelihood, log[ p(data point | latent) ]

  ## Warning: The i=i is necessary to capture the current i.
  ## Just writing
  ##   lambda sample, i: logisticModelFunction(sample, x_vals[i])
  ## will capture i by reference and thus not do what is intended.

  func_list = []
  for i in range(distrib._p.nIndividuals):
      func_list.append(lambda sample, i=i: distrib.conditional_p(sample, data_vector[i]))
  e["Problem"]["Log Likelihood Functions"] = func_list

  # # Alternative: Pass the x values to Korali. Then, the points for the individual
  # #  will be accessible to the computational model in "Data Points".
  # e["Problem"]["Log Likelihood Functions"] = [lambda sample: distrib.conditional_p(sample, internalData=True)
  #                                                   for _ in range(distrib._p.nIndividuals)]
  # e["Problem"]["Points"] = data_vector

  e["Problem"]["Latent Space Dimensions"] = distrib._p.nLatentSpaceDimensions

  e["Solver"]["Type"] = "HSAEM"
  e["Solver"]["Number Samples Per Step"] = 10
  e["Solver"]["mcmc Outer Steps"] = 1
  e["Solver"]["mcmc Target Acceptance Rate"] = 0.4
  e["Solver"]["N1"] = 2
  e["Solver"]["N2"] = 2
  e["Solver"]["N3"] = 2
  e["Solver"]["K1"] = 200
  e["Solver"]["Alpha 1"] = 0.25
  e["Solver"]["Alpha 2"] = 0.5
  e["Solver"]["Use Simulated Annealing"] = True
  e["Solver"]["Simulated Annealing Decay Factor"] = 0.95
  e["Solver"]["Simulated Annealing Initial Variance"] = 1
  e["Solver"]["Diagonal Covariance"] = True
  e["Solver"]["Termination Criteria"]["Max Generations"] = 250

  e["Distributions"][0]["Name"] = "Uniform 0"
  e["Distributions"][0]["Type"] = "Univariate/Uniform"
  e["Distributions"][0]["Minimum"] = -100
  e["Distributions"][0]["Maximum"] = 100

  e["Distributions"][1]["Name"] = "Uniform 1"
  e["Distributions"][1]["Type"] = "Univariate/Uniform"
  e["Distributions"][1]["Minimum"] = 0
  e["Distributions"][1]["Maximum"] = 100

  e["Distributions"][2]["Name"] = "Uniform 2"
  e["Distributions"][2]["Type"] = "Univariate/Uniform"
  e["Distributions"][2]["Minimum"] = 0.0
  e["Distributions"][2]["Maximum"] = 1.0

  # * Define the variables:
  #   We only define one prototype latent variable vector for individual 0.
  #   The others will be automatically generated by Korali, as well as all hyperparameters.
  if np.isscalar(distrib._p.transf):
    distrib._p.transf = [distrib._p.transf]
  if np.isscalar(distrib._p.err_transf):
    distrib._p.err_transf = [distrib._p.err_transf]
  dimCounter = 0
  distribs = {
      "Normal": "Uniform 0",
      "Log-Normal": "Uniform 1",
      "Logit-Normal": "Uniform 2",
      "Probit-Normal": "Uniform XX"
  }
  for transf in distrib._p.transf:
    generate_variable(
        transf,
        e,
        dimCounter,
        "latent parameter " + str(dimCounter),
        distribs,
        initial=distrib._p.beta[dimCounter])
    dimCounter += 1

  for i, err_transf in enumerate(distrib._p.err_transf):
    generate_variable(
        err_transf,
        e,
        dimCounter,
        "standard deviation " + str(i),
        distribs,
        initial=distrib._p.beta[dimCounter])
    dimCounter += 1

  assert dimCounter == distrib._p.dNormal + distrib._p.dLognormal + distrib._p.dLogitnormal + distrib._p.dProbitnormal

  e["File Output"]["Frequency"] = 1
  e["File Output"]["Path"] = "_korali_result_logistic_custom/"
  e["Console Output"]["Frequency"] = 1
  e["Console Output"]["Verbosity"] = "Detailed"

  k.run(e)


if __name__ == '__main__':
  # # ** For debugging, try this: **
  # import sys, trace
  # sys.stdout = sys.stderr
  # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
  # tracer.runfunc(main)
  # # ** Else: **
  main()
