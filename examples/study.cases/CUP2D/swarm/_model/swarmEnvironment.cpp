//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "swarmEnvironment.hpp"
#include <chrono>
#include <filesystem>

#define NAGENTS 9

int _argc;
char **_argv;

std::mt19937 _randomGenerator;
Simulation *_environment;

// Swimmer following an obstacle
void runEnvironment(korali::Sample &s)
{
  // Setting seed
  size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Creating results directory
  char resDir[64];
  sprintf(resDir, "%s/sample%08lu", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), sampleId);
  if( not std::filesystem::create_directories(resDir) )
  {
    fprintf(stderr, "Error creating results directory for environment\n");
    exit(-1);
  };

  // Redirecting all output to the log file
  char logFilePath[128];
  sprintf(logFilePath, "%s/log.txt", resDir);
  auto logFile = freopen(logFilePath, "a", stdout);
  if (logFile == NULL)
  {
    printf("Error creating log file: %s.\n", logFilePath);
    exit(-1);
  }

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Obtaining agents
  std::vector<StefanFish *> agents(NAGENTS);
  for( size_t i = 0; i<NAGENTS; i++ )
    agents[i] = dynamic_cast<StefanFish *>(_environment->getShapes()[i]);

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Resetting environment and setting initial conditions
  _environment->resetRL();
  for( size_t i = 0; i<NAGENTS; i++ )
    setInitialConditions(agents[i], i, s["Mode"] == "Training");

  // Setting initial state
  std::vector<std::vector<double>> states(NAGENTS);
  for( size_t i = 0; i<NAGENTS; i++ )
    states[i] = agents[i]->state();
  s["State"] = states;

  // Variables for time and step conditions
  double t = 0;        // Current time
  size_t curStep = 0;  // current Step
  double dtAct;        // Time until next action
  double tNextAct = 0; // Time of next action      

  // Setting maximum number of steps before truncation
  size_t maxSteps = 200;

  // Starting main environment loop
  bool done = false;
  while (done == false && curStep < maxSteps)
  {
    // Getting new actions
    s.update();

    // Reading new actions
    auto actions = s["Action"];

    // Setting action for each agent
    for( size_t i = 0; i<NAGENTS; i++ )
    {
      std::vector<double> action = actions[i];
      agents[i]->act(t, action);
    }

    // Run the simulation until next action is required
    dtAct = 0.;
    for( size_t i = 0; i<NAGENTS; i++ )
      if( dtAct < agents[i]->getLearnTPeriod() * 0.5 )
        dtAct = agents[i]->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;
    while ( t < tNextAct )
    {
      // Compute timestep
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;

      // Advance simulation
      _environment->advance(dt);

      // Check if there was a collision -> termination.
      done = _environment->sim.bCollision;
    }

    // Get state and action
    std::vector<double> rewards(NAGENTS);
    for( size_t i = 0; i<NAGENTS; i++ )
    {
      states[i]  = agents[i]->state();
      rewards[i] = done ? -10.0 : agents[i]->EffPDefBnd;
    }

    // Store state and action
    s["State"]  = states;
    s["Reward"] = rewards;
    
    // Printing Information:
    printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    for( size_t i = 0; i<NAGENTS; i++ )
    {
      auto state = states[i];
      printf("[Korali] AGENT %ld\n", i);
      printf("[Korali] State: [ %.3f", state[0]);
      for (size_t j = 1; j < state.size(); j++) printf(", %.3f", state[j]);
      printf("]\n");
      printf("[Korali] Action: [ %.3f, %.3f ]\n", actions[i][0], actions[i][1]);
      printf("[Korali] Reward: %.3f\n", rewards[i]);
      printf("[Korali] Terminal?: %d\n", done);
      printf("[Korali] -------------------------------------------------------\n");
    }
    fflush(stdout);

    // Advancing to next step
    curStep++;
  }

  // Setting finalization status
  if (done == true)
    s["Termination"] = "Terminal";
  else
    s["Termination"] = "Truncated";

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

  // Closing log file
  fclose(logFile);
}

std::vector<std::vector<double>> initialPositions{
    {0.20, 1.00},
    {0.50, 0.90},
    {0.50, 1.10},
    {0.80, 0.80},
    {0.80, 1.00},
    {0.80, 1.20},
    {1.10, 0.90},
    {1.10, 1.10},
    {1.40, 1.00} };

void setInitialConditions(StefanFish *agent, size_t agentId, const bool isTraining)
{
  // Initial fixed conditions
  double initialAngle = 0.0;
  auto initialPosition = initialPositions[agentId];

  // or with noise
  //if (isTraining)
  //{
  // std::uniform_real_distribution<double> disA(-20. / 180. * M_PI, 20. / 180. * M_PI);
  // std::uniform_real_distribution<double> disX(-0.25, 0.25);
  // std::uniform_real_distribution<double> disY(-0.125, 0.125);

  // initialAngle = disA(_randomGenerator);
  // initialPosition[0] = initialPosition[0] + disX(_randomGenerator);
  // initialPosition[1] = initialPosition[1] + disY(_randomGenerator);
  //}

  printf("[Korali] Initial Conditions:\n");
  printf("[Korali] SA: %f\n", initialAngle);
  printf("[Korali] SX: %f\n", initialPosition[0]);
  printf("[Korali] SY: %f\n", initialPosition[1]);

  // Setting initial position and orientation for the fish
  agent->setCenterOfMass(initialPosition.data());
  agent->setOrientation(initialAngle);

  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();
}
