#!/usr/bin/env python3
from tictaktoe import *
import numpy as np

######## Defining Environment Storage

tictaktoe_game = TicTakToeGame()
maxSteps = 500

def env(s):

 # Initializing environment and random seed
 sampleId = s["Sample Id"]
 launchId = s["Launch Id"]

 tictaktoe_game.reset(sampleId * 1024 + launchId)
 s["State"] = tictaktoe_game.getState().tolist()
 #print(s["State"])

 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  done = tictaktoe_game.advance(s["Action"])
  #print("action {}".format(s["Action"])) 
  
  # Getting Reward
  s["Reward"] = tictaktoe_game.getReward()
   
  # Storing New State
  s["State"] = tictaktoe_game.getState().tolist()
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (tictaktoe_game.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
