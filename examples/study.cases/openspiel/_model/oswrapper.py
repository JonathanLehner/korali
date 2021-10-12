#!/usr/bin/env python3

##  This code is a wrapper for Openspiel

##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.

import math
import numpy as np, sys
from scipy.integrate import ode
import pyspiel

class OSWrapper:
  def __init__(self, envName):
    # should be "leduc_poker" or "tic_tac_toe" etc.
    print("Creating game: {}".format(envName))
    self.game = pyspiel.load_game(envName)
    self.game_state = self.game.new_initial_state()
    self.is_failed = False
    self.deal_chance()
    print(self.game_state)

  def reset(self, seed):
    np.random.seed(seed)
    self.is_failed = False
    self.game_state = self.game.new_initial_state()
    self.deal_chance()

  def isFailed(self):
    return self.is_failed

  def isOver(self):
    return self.game_state.is_terminal()

  def advance(self, action):
    # the legal actions depend on the game state
    # for instance in poker the available bet sizes depend on the remaining stack of the active player
    legal_actions = self.game_state.legal_actions()
    action = int(action[0])
    if(action not in legal_actions):
      self.is_failed = True
      return 1 # terminate when illegal action

    self.game_state.apply_action(action)
    self.deal_chance()
    
    if self.isOver(): 
      return 1
    else: 
      return 0

  def deal_chance(self):
    while self.game_state.is_chance_node():
      random_action = np.random.choice(self.game_state.legal_actions())
      self.game_state.apply_action(random_action)

  def getActiveState(self):
    cur_player = self.game_state.current_player()
    # the state of -4 means game is over
    # sometimes the state is read after the game is over and then cur_player < 0
    if(cur_player < 0): 
      cur_player = 0 

    info_state = self.game_state.observation_tensor(cur_player)
    info_state = np.asarray(info_state)

    return info_state

  def getState(self, i):
    print("getState")
    info_state = self.game_state.observation_tensor(i)
    info_state = np.asarray(info_state)
    print("getState2")
    return info_state

  @property
  def observation_space(self):
    return self.getState(0)

  @property
  def num_players(self):
    return self.game_state.num_players()

  @property
  def action_space(self):
    # openspiel does not supply all actions, so we assume that the intiai state contains all actions
    # this might not be the case for all games
    available_actions = self.game_state.legal_actions()
    return available_actions

  def getReward(self, i):
    if(self.isFailed()):
      return -100 # illegal moves
    elif(self.game_state.is_terminal()):
      rewards = self.game_state.rewards() # reward for winning player, current player at terminal state is always -4
      return rewards[i] # reward the player who made the winning move i.e. made the game terminal
    else: 
      return 1 # small reward until game is over - we do not know in Leduc if the current state is good until the end of the game
