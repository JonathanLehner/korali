#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *
import argparse

parser = argparse.ArgumentParser()
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

### Defining the Leduc game's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Discrete"
e["Problem"]["Possible Actions"] = [ [0], [1], [2], [3], [4], [5] ]
e["Problem"]["Environment Function"] = env
e["Problem"]["Training Reward Threshold"] = 300
e["Problem"]["Policy Testing Episodes"] = 40
e["Problem"]["Actions Between Policy Updates"] = 5

e["Variables"][0]["Name"] = "Current player"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Pot"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Money player 1"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Money player 2"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Public card" # -1 in first round
e["Variables"][4]["Type"] = "State"

e["Variables"][5]["Name"] = "Private card current player"
e["Variables"][5]["Type"] = "State"

e["Variables"][6]["Name"] = "Round 1 action 1"
e["Variables"][6]["Type"] = "State"

e["Variables"][7]["Name"] = "Round 1 action 2"
e["Variables"][7]["Type"] = "State"

e["Variables"][8]["Name"] = "Round 2 action 1"
e["Variables"][8]["Type"] = "State"

e["Variables"][9]["Name"] = "Round 2 action 2"
e["Variables"][9]["Type"] = "State"

e["Variables"][10]["Name"] = "Move"
e["Variables"][10]["Type"] = "Action"

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Discrete / dVRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = float(args.learningRate)
e["Solver"]["Mini Batch"]["Size"] = 32
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
e["Solver"]["Reward"]["Rescaling"]["Frequency"] = 1000
  
### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = args.engine
e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

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

