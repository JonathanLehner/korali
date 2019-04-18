#!/usr/bin/env python
import sys
sys.path.append('/home/martiser/skorali/korali/lib')
import libkorali

korali = libkorali.Engine()

korali["Verbosity"] = "Normal";

for i in range(4):
 korali["Parameters"][i]["Name"] = "X" + str(i);
 korali["Parameters"][i]["Distribution"] = "Uniform";
 korali["Parameters"][i]["Type"] = "Computational";
 korali["Parameters"][i]["Minimum"] = -32.0;
 korali["Parameters"][i]["Maximum"] = +32.0;

korali["Problem"]["Objective"] = "Direct Evaluation";
korali["Solver"]["Method"] = "CMA-ES";
korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-11;
korali["Solver"]["Termination Criteria"]["Max Generations"] = 600;
korali["Solver"]["Lambda"] = 128;

korali.run()