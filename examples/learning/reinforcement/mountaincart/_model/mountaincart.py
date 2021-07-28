#!/usr/bin/env python3

##  This code was inspired by the OpenAI Gym Mountaincar-v0 environment

## Track described by function x^2

##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.

import math
import numpy as np, sys
from scipy.integrate import ode

class MountainCart:
  def __init__(self):
    self.dt = 0.1
    self.ddt = 0.01
    self.step=0
    self.t=0
    self.u = np.asarray([0, 0, 0, 0, 0, 0]) # location x, location y, velocity x, velocity y, acceleration x, acceleration y
    self.maxForce = 1.0 # max absolute force
    self.F = 5.0 # force in cart direction (facing right)
    self.m = 1.0 # mass
    self.g = 9.81 # gravitaty
    self.highest = 0

    assert(self.g > self.maxForce/self.m) # avoid weird configuration

  def reset(self, seed):
    np.random.seed(seed)
    
    self.step = 0
    self.F = 0
    self.t = 0
 
    self.u = np.random.uniform(-0.05, 0.05, 6)
    self.highest = max(self.highest, self.u[1])
    
    slope = 2.*self.u[0] # track is x^2
    theta = np.arctan(slope)
    f = np.sin(theta)*self.m*self.g
    fxg = np.sin(theta)*f
    fyg = np.cos(theta)*f

    # reset acceleration
    self.u[4] = (fxg+fxc)/self.m
    self.u[5] = (fyg+fyc)/self.m

  def isFailed(self):
    return False

  def isOver(self):
    return self.isFailed()

  def advance(self, action):
    self.F = action[0]
    self.F = np.clip(self.F, a_min=-self.maxForce, a_max=+self.maxForce)
    
    # simulate dt seconds 
    for i in range(int(self.dt/self.ddt)):
        # leapfrog scheme (location)
        self.u[0] = self.u[0] + self.u[2]*self.dt + 0.5*self.u[4]*self.dt**2
        self.u[1] = self.u[0]**2 # track is x^2 # self.u[1] + self.u[3]*self.dt + 0.5*self.u[5]*self.dt**2
        
        # leapfrog scheme (velocity part a)
        self.u[2] = self.u[2] + 0.5*self.u[4]*self.dt
        self.u[3] = self.u[3] + 0.5*self.u[5]*self.dt
     
        # update acceleration at new position
        slope = 2.*self.u[0] # "ground" is x^2
        theta = np.arctan(slope)
        fg = np.sin(theta)*self.m*self.g
        fxg = np.sin(theta)*fg
        fyg = np.cos(theta)*fg

        fxc = np.cos(theta)*self.F
        fyc = np.sin(theta)*self.F

        # update acceleration
        self.u[4] = (fxg+fxc)/self.m
        self.u[5] = (fyg+fyc)/self.m

        # leapfrog scheme (velocity part b)
        self.u[2] = self.u[2] + 0.5*self.u[4]*self.dt
        self.u[3] = self.u[3] + 0.5*self.u[5]*self.dt
        
        self.t = self.t + self.ddt
    
    self.step = self.step + 1
    
    if self.isOver(): 
      return 1
    else: 
      return 0

  def getState(self):
    state = np.zeros(4)
    state[0] = self.u[0] # Cart Position x
    state[1] = self.u[1] # Cart Position y
    state[2] = self.u[2] # Cart Velocity x
    state[3] = self.u[3] # Cart Velocity y
    return state

  def getReward(self):
    if self.u[1] > self.highest:
        self.highest = self.u[1]
        return self.highest
    else:
        return 0