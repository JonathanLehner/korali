#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "-m", "korali.rlview", "--help"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.rlview", "--dir", "abf2d_vracer1", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.rlview", "--dir", "abf2d_vracer1", "--maxObservations", "10000", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.rlview", "--dir", "abf2d_vracer1", "--maxReward", "20.0", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.rlview", "--dir", "abf2d_vracer1", "--minReward", "-1.0", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.rlview", "--dir", "abf2d_vracer1", "--updateFrequency", "1.0", "--maxPlottingTime", "5", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.rlview", "--dir", "abf2d_vracer1", "--showCI", "0.2", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.rlview", "--dir", "abf2d_vracer1", "--averageDepth", "30", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.rlview", "--dir", "abf2d_vracer1", "abf2d_vracer2", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.rlview", "--dir", "abf2d_vracer1", "abf2d_vracer2", "--aggregate", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.rlview", "--dir", "abf2d_vracer1", "--output", "test.png", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.rlview", "--dir", "abf2d_vracer1", "abf2d_vracer2", "--test", "--maxObservations", "10000", "--maxReward", "20.0", "--minReward", "-1.0", "--updateFrequency", "1.0",  "--showCI", "0.2", "--averageDepth", "30", "--maxPlottingTime", "5",])
if r!=0:
  exit(r)


exit(0)
