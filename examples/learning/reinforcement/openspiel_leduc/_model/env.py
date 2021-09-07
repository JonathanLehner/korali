#!/usr/bin/env python3
from leduc import *
import numpy as np

######## Defining Environment Storage

leduc_game = Leduc()
maxSteps = 500

def env(s):

 # Initializing environment and random seed
 sampleId = s["Sample Id"]
 launchId = s["Launch Id"]

 leduc_game.reset(sampleId * 1024 + launchId)
 s["State"] = leduc_game.getState().tolist()
 #print(s["State"])

 step = 0
 done = False

 while not done and step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  done = leduc_game.advance(s["Action"])
  #print("action {}".format(s["Action"])) 
  
  # Getting Reward
  s["Reward"] = leduc_game.getReward()
   
  # Storing New State
  s["State"] = leduc_game.getState().tolist()
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (leduc_game.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"
