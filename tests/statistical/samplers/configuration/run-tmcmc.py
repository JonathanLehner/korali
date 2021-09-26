#!/usr/bin/env python3

# Importing computational model
import sys
import numpy as np

def verify(actual, expected, atol=1e-9):
  assert np.isclose(
        expected, actual, atol=atol
    ), "Value {} deviates from expected value {}".format(actual, expected)

def lgaussian(s):
  x0 = s["Parameters"][0]
  x1 = s["Parameters"][1]
  r = - 0.5* (x0*x0 + x1*x1)
  s["logLikelihood"] = r

# Starting Korali's Engine
import korali
k = korali.Engine()
e = korali.Experiment()

# Setting up custom likelihood for the Bayesian Problem
e["Problem"]["Type"] = "Bayesian/Custom"
e["Problem"]["Likelihood Model"] = lgaussian

# Configuring TMCMC parameters
e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 100

# Configuring the problem's random distributions
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = 0.0
e["Distributions"][0]["Maximum"] = 5.0

# Configuring the problem's variables and their prior distributions
e["Variables"][0]["Name"] = "a"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"

e["Variables"][1]["Name"] = "b"
e["Variables"][1]["Prior Distribution"] = "Uniform 0"

e["Solver"]["Termination Criteria"]["Max Generations"] = 1

# Running Korali
e["File Output"]["Enabled"] = False
k.run(e)

# Verify Initial Values
verify(e["Solver"]["Chain Candidates LogPriors"][0], np.log(1./25.))
verify(e["Solver"]["Chain Leaders LogPriors"][0], np.log(1./25.))
