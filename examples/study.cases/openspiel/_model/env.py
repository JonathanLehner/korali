#!/usr/bin/env python3
from oswrapper import *
import numpy as np

######## Defining Environment Storage
maxSteps = 500

def agent(s, env):

 # Initializing environment and random seed
 sampleId = s["Sample Id"]
 launchId = s["Launch Id"]

 env.reset(sampleId * 1024 + launchId)
 num_players = env.num_players

 states = []
 for i in np.arange(num_players):
  # get state
  state = env.getState()
  states.append(state)
 s["State"] = states

 #s["State"] = env.getState().tolist()
 #print(s["State"])

 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  # player is automatically switching after advance
  done = env.advance(s["Action"])
  #print("action {}".format(s["Action"])) 
  
  # Getting Reward
  s["Reward"] = env.getReward()
   
  # Storing New State
  s["State"] = env.getState().tolist()
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (env.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"


def initEnvironment(e, envName, moviePath = ''):

 # Creating environment 
 env = OSWrapper(envName)
   
 ### Defining problem configuration for openAI Gym environments
 e["Problem"]["Type"] = "Reinforcement Learning / Discrete"
 e["Problem"]["Environment Function"] = lambda x : agent(x, env)
 e["Problem"]["Custom Settings"]["Print Step Information"] = "Disabled"
 
 # Getting environment variable counts
 stateVariableCount = len(env.observation_space)
 actionVariableCount = len(env.action_space)
 
 # Generating state variable index list
 stateVariablesIndexes = range(stateVariableCount)
   
 # Defining State Variables
 for i in stateVariablesIndexes:
  e["Variables"][i]["Name"] = "State Variable " + str(i)
  e["Variables"][i]["Type"] = "State"
  
 # Defining Action Variables
 e["Variables"][stateVariableCount]["Name"] = "Move"
 e["Variables"][stateVariableCount]["Type"] = "Action"
 e["Problem"]["Possible Actions"] = [[x] for x in range(actionVariableCount)]