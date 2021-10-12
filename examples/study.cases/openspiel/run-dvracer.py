#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)
parser.add_argument(
    '--engine',
    help='NN backend to use',
    default='OneDNN',
    required=False)
parser.add_argument(
    '--maxGenerations',
    help='Maximum Number of generations to run',
    default=1000,
    required=False)    
parser.add_argument(
    '--optimizer',
    help='Optimizer to use for NN parameter updates',
    default='Adam',
    required=False)
parser.add_argument(
    '--learningRate',
    help='Learning rate for the selected optimizer',
    default=1e-3,
    required=False)
parser.add_argument(
    '--concurrentEnvironments',
    help='Number of environments to run concurrently',
    default=1,
    required=False)
parser.add_argument(
    '--testRewardThreshold',
    help='Threshold for the testing MSE, under which the run will report an error',
    default=150,
    required=False)
args = parser.parse_args()

print("Running Leduc example with arguments:")
print(args)

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Initializing Openspiel environment

initEnvironment(e, args.env)

### Defining the Openspiel game's configuration
e["Problem"]["Training Reward Threshold"] = 300
e["Problem"]["Policy Testing Episodes"] = 40
e["Problem"]["Actions Between Policy Updates"] = 5

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Discrete / dVRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = float(args.learningRate)
e["Solver"]["Mini Batch"]["Size"] = 256
e["Solver"]["Concurrent Environments"] = int(args.concurrentEnvironments)

### Defining Experience Replay configuration

e["Solver"]["Experience Replay"]["Start Size"] = 4096
e["Solver"]["Experience Replay"]["Maximum Size"] = 65536

### Setting Experience Replay and REFER settings

e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1

e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
  
### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Generations"] = args.maxGenerations
e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 250

### Setting file output configuration

e["File Output"]["Enabled"] = False
e["Console Output"]["Verbosity"] = "Detailed"

### Running Experiment

k.run(e)

### Now we run a few test samples and check their reward

e["Solver"]["Mode"] = "Testing"
e["Solver"]["Testing"]["Sample Ids"] = list(range(5))

k.run(e)

averageTestReward = np.average(e["Solver"]["Testing"]["Reward"])
print("Average Reward: " + str(averageTestReward))
if (averageTestReward < 100):
 print("Tic-Tak-Toe example did not reach minimum testing average.")
 exit(-1)

