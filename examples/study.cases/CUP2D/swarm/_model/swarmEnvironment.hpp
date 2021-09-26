//  Korali environment for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>

void runEnvironment(korali::Sample &s);
extern std::string _resultsPath;
extern int _argc;
extern char **_argv;

#include "Obstacles/StefanFish.h"
#include "Simulation.h"

void initializeEnvironment();
void setInitialConditions(StefanFish *agent, size_t agentId, const bool isTraining);

// Global variables for the simulation (ideal if this would be a class instead)
extern std::mt19937 _randomGenerator;
extern Simulation *_environment;
