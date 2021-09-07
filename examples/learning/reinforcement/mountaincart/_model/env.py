#!/usr/bin/env python3
from mountaincart import *

import pickle

# generate states file 'states.pickle'?
output = False

######## Defining Environment Storage

cart = MountainCart()
maxSteps = 1000 

def env(s):

 # Initializing environment and random seed
 sampleId = s["Sample Id"]
 launchId = s["Launch Id"]
 cart.reset(sampleId * 1024 + launchId)
 s["State"] = cart.getState().tolist()
 step = 0
 done = False

 while step < maxSteps:

  # Getting new action
  s.update()
  
  # Performing the action
  done = cart.advance(s["Action"])
  
  # Getting Reward
  s["Reward"] = cart.getReward()
   
  # Storing New State
  s["State"] = cart.getState().tolist()
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (cart.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"

 # Generate output file with states and actions
 if output:

     data = {
	'action': cart.actions,
	'location': cart.locations,
	'velocity': cart.velocity,
	'acceleration' : cart.acceleration,
	'fgravity' : cart.fgravity
     }

     with open('states.pickle', 'wb') as fp:
       pickle.dump(data, fp)


