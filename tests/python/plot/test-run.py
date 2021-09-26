#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "-m", "korali.plot", "--help"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.plot", "--dir", "cmaes", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.plot", "--dir", "dea", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.plot", "--dir", "mocmaes", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.plot", "--dir", "lmcmaes", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.plot", "--dir", "tmcmc", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.plot", "--dir", "mcmc", "--test"])
if r!=0:
  exit(r)

exit(0)
