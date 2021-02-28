#!/usr/bin/env python3
import os
import sys
import argparse
sys.path.append('_model')
from agent import *

####### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)
args = parser.parse_args()

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

resultFolder = '_result_ddpg/' 
e.loadState(resultFolder + '/latest');

### Initializing openAI Gym environment

initEnvironment(e, args.env)

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / DDPG"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 1
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Policy"]["Learning Rate Scale"] = 0.1
e["Solver"]["Policy"]["Adoption Rate"] = 0.01

### Defining the configuration of replay memory

e["Solver"]["Experience Replay"]["Start Size"] = 32768
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144

### Configuring the Remember-and-Forget Experience Replay algorithm

e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5e-7
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3

### Configuring Mini Batch

e["Solver"]["Mini Batch Size"] = 128
e["Solver"]["Mini Batch Strategy"] = "Uniform"

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Setting console/file output configuration

e["Solver"]["Termination Criteria"]["Max Generations"] = 100000
e["Solver"]["Experience Replay"]["Serialize"] = True
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 5
e["File Output"]["Path"] = resultFolder

### Running Experiment

k.run(e)
