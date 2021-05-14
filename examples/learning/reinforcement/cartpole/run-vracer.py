#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Training Reward Threshold"] = 400
e["Problem"]["Policy Testing Episodes"] = 20
e["Problem"]["Actions Between Policy Updates"] = 5

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Cart Velocity"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Pole Angle"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Pole Angular Velocity"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Force"
e["Variables"][4]["Type"] = "Action"
e["Variables"][4]["Lower Bound"] = -10.0
e["Variables"][4]["Upper Bound"] = +10.0
e["Variables"][4]["Initial Exploration Noise"] = 1.0

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 10
e["Solver"]["Episodes Per Generation"] = 1
e["Solver"]["Policy"]["Distribution"] = "Normal"

e["Solver"]["Experience Replay"]["Start Size"] = 1000
e["Solver"]["Experience Replay"]["Maximum Size"] = 10000

e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Learning Rate"] = 1e-3
e["Solver"]["Mini Batch"]["Size"] = 32

e["Solver"]["State Rescaling"]["Enabled"] = False
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = False
e["Solver"]["Reward"]["Rescaling"]["Frequency"] = 1000

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "AdaBelief"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Generations"] = 1000
e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 450

### Setting file output configuration

e["File Output"]["Enabled"] = True

### Running Experiment

k.run(e)
