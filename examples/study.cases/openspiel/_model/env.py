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
  # get states
  state = env.getState(i).tolist()
  states.append(state)
 s["State"] = states

 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new simultaneous actions for all players in turn 
  # --> for most games only one player makes a turn and the others only special non-actions
  s.update()

  # Performing the action
  # player is automatically switching after advance
  actions = s["Action"]
  for i in np.arange(num_players):
    # could also consider to only let the active player make a move
    # instead of the waiting action
    # but this would need a check of the environment
    done = env.advance(actions[i], i)
  
  # set state
  states  = []
  rewards = []
  for i in np.arange(num_players):
    # get state
    state = env.getState(i).tolist()
    states.append(state)
    # get reward
    reward = env.getReward(i)
    rewards.append(reward)

  s["State"] = states
  if(min(rewards) < 0):
    print(rewards)
  s["Reward"] = rewards
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (env.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"


def initEnvironment(e, envName):

 # Creating environment 
 env = OSWrapper(envName)
   
 e["Problem"]["Agents Per Environment"] = env.num_players

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

 possible_actions = env.possible_actions
 actions = [[possible_actions[x]] for x in range(actionVariableCount)]
 actions.append([-888]) # add the "waiting" action for players when it is not their turn
 e["Problem"]["Possible Actions"] = actions