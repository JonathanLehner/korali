#!/usr/bin/env python3

##  This code was inspired by the OpenAI Gym CartPole v0 environment

##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.

import math
import numpy as np, sys
from scipy.integrate import ode
import pyspiel

class Leduc:
  def __init__(self):
    game_string = "leduc_poker"
    print("Creating game: {}".format(game_string))
    self.game = pyspiel.load_game(game_string)
    self.game_state = self.game.new_initial_state()
    self.is_failed = False
    self.deal_cards()
    print(self.game_state)

  def reset(self, seed):
    np.random.seed(seed)
    self.is_failed = False
    self.game_state = self.game.new_initial_state()
    self.deal_cards()

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
    self.deal_cards()
    
    if self.isOver(): 
      return 1
    else: 
      return 0

  def deal_cards(self):
    while self.game_state.is_chance_node():
      random_action = np.random.choice(self.game_state.legal_actions())
      self.game_state.apply_action(random_action)

  def getState(self):
    state = np.ones(10)*-1
    cur_player = self.game_state.current_player()
    if(cur_player < 0):
      cur_player = 0 # should not get the state at -4, which means game is over

    state[0] = cur_player # Current player    
    info_state = self.game_state.information_state_string(cur_player)
    # pot
    state[1] = int(info_state.split("][")[0].split(": ")[1])
    # money player 1 
    state[2] = int(info_state.split("][")[5].split(": ")[1].split(" ")[0])
    # money player 2
    state[3] = int(info_state.split("][")[5].split(": ")[1].split(" ")[1])
    # public card
    if(len(info_state.split("][")) == 9):
      state[4] = -1
      pk = info_state.split("][")[6].split(": ")[1]
      if(pk != ""):
         state[4] = int(pk)
      mr1 = info_state.split("][")[7].split(": ")[1] 
      mr2 = info_state.split("][")[8].split(": ")[1][:-1]
    else:
      mr1 = info_state.split("][")[6].split(": ")[1] 
      mr2 = info_state.split("][")[7].split(": ")[1][:-1]

    # private card current player
    state[5] = int(info_state.split("][")[1].split(": ")[1])
   
    # moves round 1
    state[6] = -1
    state[7] = -1
    if(mr1 != ""): # can there be a third bet?
      bets = mr1.split(" ")
      state[6] = int(bets[0])
      if(len(bets) > 1):
        state[7] = int(bets[1])

    # moves round 2 
    state[8] = -1
    state[9] = -1
    if(mr2 != ""): # can there be a third bet?
      bets = mr2.split(" ")
      state[8] = int(bets[0])
      if(len(bets) > 1):
        state[9] = int(bets[1])
      
    return state


  def getReward(self):
    if(self.isFailed()):
      return -100 # illegal moves
    elif(self.game_state.is_terminal()):
      rewards = max(self.game_state.rewards()) # reward for winning player, current player at terminal state is always -4
      return rewards # reward the player who made the winning move i.e. made the game terminal
    else: 
      return 0 # no reward until game is over
