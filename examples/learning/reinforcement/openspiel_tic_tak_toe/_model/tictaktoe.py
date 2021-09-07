#!/usr/bin/env python3

##  This code was inspired by the OpenAI Gym CartPole v0 environment

##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.

import math
import numpy as np, sys
from scipy.integrate import ode
import pyspiel

class TicTakToeGame:
  def __init__(self):
    game_string = "tic_tac_toe"
    print("Creating game: {}".format(game_string))
    self.game = pyspiel.load_game(game_string)
    self.game_state = self.game.new_initial_state()
    self.is_failed = False
    print(self.game_state)

  def reset(self, seed):
    np.random.seed(seed)
    self.is_failed = False
    self.game_state = self.game.new_initial_state()

  def isFailed(self):
    return self.is_failed

  def isOver(self):
    return self.game_state.is_terminal()

  def advance(self, action):
    legal_actions = self.game_state.legal_actions()
    action = int(action[0])
    if(action not in legal_actions):
      self.is_failed = True
      return 1 # terminate when illegal action

    self.game_state.apply_action(action)
    
    if self.isOver(): 
      return 1
    else: 
      return 0

  def getState(self):
    state = np.ones(10)*-1
    state[0] = self.game_state.current_player() # Current player    
    if(self.game_state.information_state_string(0) != ''):
      actions = self.game_state.information_state_string(0).split(", ")
      for i in range(0, len(actions)):
        state[i+1] = int(actions[i])

    return state

  def getReward(self):
    if(self.isFailed()):
      return -100 # illegal moves
    elif(self.game_state.is_terminal()):
      rewards = max(self.game_state.rewards()) # reward for winning player, current player at terminal state is always -4
      return rewards # reward the player who made the winning move i.e. made the game terminal
    else: 
      return 0 # no reward until game is over