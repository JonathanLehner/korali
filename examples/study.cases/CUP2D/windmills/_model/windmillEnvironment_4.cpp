#include "windmillEnvironment.hpp"
#include <chrono>
#include <filesystem>

int _argc;
char **_argv;

Simulation *_environment;
std::mt19937 _randomGenerator;

// 4 windmills with variable torque applied to them
void runEnvironment(korali::Sample &s)
{
  ////////////////////////////////////////// setup stuff 
  // Setting seed
  size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Creating results directory
  char resDir[64];
  sprintf(resDir, "%s/sample%08lu", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), sampleId);
  std::filesystem::create_directories(resDir);

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
  ////////////////////////////////////////// setup stuff 


  ////////////////////////////////////////// Initialize agents and objective
  // Obtaining agent, 4 windmills 
  Windmill* agent1 = dynamic_cast<Windmill*>(_environment->getShapes()[0]);
  Windmill* agent2 = dynamic_cast<Windmill*>(_environment->getShapes()[1]);
  Windmill* agent3 = dynamic_cast<Windmill*>(_environment->getShapes()[2]);
  Windmill* agent4 = dynamic_cast<Windmill*>(_environment->getShapes()[3]);

  // useful agent functions :
  // void act( double action );
  // double reward( std::array<Real,2> target, std::vector<double> target_vel, double C = 10);
  // std::vector<double> state();

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Reseting environment and setting initial conditions
  _environment->resetRL();
  bool random_init = (s["Mode"] == "Training");
  setInitialConditions(agent1, 0, random_init);
  setInitialConditions(agent2, 0, random_init);
  setInitialConditions(agent3, 0, random_init);
  setInitialConditions(agent4, 0, random_init);

  // Set target
  std::array<Real,2> target_pos{0.7,0.5};
  std::vector<double> target_vel={0.0,0.0};

  std::vector<double> state1 = agent1->state();
  std::vector<double> state2 = agent2->state();
  std::vector<double> state3 = agent3->state();
  std::vector<double> state4 = agent4->state();

  std::vector<double> state = {state1[0], state1[1], state2[0], state2[1], state3[0], state3[1], state4[0], state4[1]};

  s["State"] = state;

  // Setting initial time and step conditions
  double t = 0;        // Current time
  double tNextAct = 0; // Time until next action
  size_t curStep = 0;  // current Step

  // Setting maximum number of steps before truncation
  size_t maxSteps = 2000;

  // Starting main environment loop
  // bool done = false;

  while (curStep < maxSteps)
  {

 // Getting initial time
    auto beginTime = std::chrono::steady_clock::now(); // Profiling

    // Getting new action
    s.update();

    // Reading new action
    std::vector<double> action = s["Action"];

    // Setting action
    agent1->act( action[0] );
    agent2->act( action[1] );
    agent3->act( action[2] );
    agent4->act( action[3] );

    // Run the simulation until next action is required
    tNextAct += 0.01;
    while ( t < tNextAct )
    {
      // Advance simulation
      const double dt = _environment->calcMaxTimestep();
      t += dt;

      // Advance simulation and check whether it is correct
      if (_environment->advance(dt))
      {
        fprintf(stderr, "Error during environment\n");
        exit(-1);
      }
    }

    // reward( std::vector<double> target, std::vector<double> target_vel, double C = 10)
    double C = 0.5e8;
    double r1 = agent1->reward( target_pos, target_vel,  C);
    double r2 = agent2->reward( target_pos, target_vel,  C);
    double r3 = agent3->reward( target_pos, target_vel,  C);
    double r4 = agent4->reward( target_pos, target_vel,  C);
    double reward = (r1 + r2 + r3 + r4);

    // Getting ending time
    auto endTime = std::chrono::steady_clock::now(); // Profiling
    double actionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count() / 1.0e+9;

    // Obtaining new agent state
    state1 = agent1->state();
    state2 = agent2->state();
    state3 = agent3->state();
    state4 = agent4->state();
    state = {state1[0], state1[1], state2[0], state2[1], state3[0], state3[1], state4[0], state4[1]};

    // Printing Information:
    printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    printf("[Korali] State: [ ");
    for (size_t i = 0; i < state.size(); i++){
      if (i%2 == 0){
        printf("[%.3f, ", state[i]);
      } else {
        printf("%.3f]", state[i]);
      }
    }
    printf("]\n");
    printf("[Korali] Previous Torque: [ %.8f, %.8f, %.8f, %.8f  ]\n", action[0], action[1], action[2], action[3]);
    printf("[Korali] Reward: %.3f\n", reward);
    printf("[Korali] Time: %.3fs\n", actionTime);
    printf("[Korali] -------------------------------------------------------\n");
    fflush(stdout);

    // Storing reward
    s["Reward"] = reward;

    // Storing new state
    s["State"] = state;

    // Advancing to next step
    curStep++;
  }

  // Setting finalization status
  // if (done == true)
  //   s["Termination"] = "Terminal";
  // else
  //   s["Termination"] = "Truncated";
  s["Termination"] = "Truncated";

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

  // Closing log file
  fclose(logFile);
}

// set initial conditions of the agent
void setInitialConditions(Windmill* agent, double init_angle, bool randomized)
{
  // Intial fixed condition of angle and angular velocity
  double angle = init_angle;

  // set random beginning angle
  if(randomized)
  {
    // windmills have 3 axis of symmetry, meaning if we rotate it by 120°,we have the same setup
    std::uniform_real_distribution<double> dis(0, 2*(M_PI/3));
    angle = dis(_randomGenerator);
  }
  
  printf("[Korali] Initial Conditions:\n");
  printf("[Korali] orientation: %f\n", angle);

  agent->setOrientation(angle);

  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();
}

// bool isTerminal(SmartCylinder* agent, std::vector<double>& target )
// {
//   const double dX = (agent->center[0] - target[0]);
//   const double dY = (agent->center[1] - target[1]);

//   const double dTarget = std::sqrt(dX*dX+dY*dY);

//   bool terminal = false;
//   if ( dTarget < 1e-1 ) terminal = true;

//   return terminal;
// }


  // // Obtaining agent
  // SmartCylinder* agent = dynamic_cast<SmartCylinder *>(_environment->getShapes()[0]);

  // // Establishing environment's dump frequency
  // _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // // Reseting environment and setting initial conditions
  // _environment->resetRL();
  // std::vector<double> start{0.2,0.5};
  // setInitialConditions(agent, start, s["Mode"] == "Training");

  // // Set target 
  // std::vector<double> target{0.8,0.5};

  // // Setting initial state
  // auto state = agent->state( target );
  // s["State"] = state;

  // // Setting initial time and step conditions
  // double t = 0;        // Current time
  // double tNextAct = 0; // Time until next action
  // size_t curStep = 0;  // current Step

  // // Setting maximum number of steps before truncation
  // size_t maxSteps = 200;

  // // Starting main environment loop
  // bool done = false;
  // while (done == false && curStep < maxSteps)
  // {
  //   // Getting initial time
  //   auto beginTime = std::chrono::steady_clock::now(); // Profiling

  //   // Getting new action
  //   s.update();

  //   // Reading new action
  //   std::vector<double> action = s["Action"];

  //   // Setting action
  //   agent->act( action );

  //   // Run the simulation until next action is required
  //   tNextAct += 0.1;
  //   while ( t < tNextAct )
  //   {
  //     // Advance simulation
  //     const double dt = _environment->calcMaxTimestep();
  //     t += dt;

  //     // Advance simulation and check whether it is correct
  //     if (_environment->advance(dt))
  //     {
  //       fprintf(stderr, "Error during environment\n");
  //       exit(-1);
  //     }

  //     // Re-check if simulation is done.
  //     done = isTerminal( agent, target );
  //   }

  //   // Reward is +10 if state is terminal; otherwise obtain it from inverse distance to target
  //   double reward = done ? 100.0 : agent->reward( target );

  //   // Getting ending time
  //   auto endTime = std::chrono::steady_clock::now(); // Profiling
  //   double actionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count() / 1.0e+9;

  //   // Printing Information:
  //   printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
  //   printf("[Korali] State: [ %.3f", state[0]);
  //   for (size_t i = 1; i < state.size(); i++) printf(", %.3f", state[i]);
  //   printf("]\n");
  //   printf("[Korali] Force: [ %.3f, %.3f ]\n", action[0], action[1]);
  //   printf("[Korali] Reward: %.3f\n", reward);
  //   printf("[Korali] Terminal?: %d\n", done);
  //   printf("[Korali] Time: %.3fs\n", actionTime);
  //   printf("[Korali] -------------------------------------------------------\n");
  //   fflush(stdout);

  //   // Obtaining new agent state
  //   state = agent->state( target );

  //   // Storing reward
  //   s["Reward"] = reward;

  //   // Storing new state
  //   s["State"] = state;

  //   // Advancing to next step
  //   curStep++;
  // }

  // // Setting finalization status
  // if (done == true)
  //   s["Termination"] = "Terminal";
  // else
  //   s["Termination"] = "Truncated";

  // // Switching back to experiment directory
  // std::filesystem::current_path(curPath);

  // // Closing log file
  // fclose(logFile);
  // }

// void setInitialConditions(SmartCylinder* agent, std::vector<double>& start, bool randomized)
// {
//   // Initial fixed conditions
//   double locationX = start[0];
//   double locationY = start[1];

//   // or with noise
//   if (randomized)
//   {
//     std::uniform_real_distribution<double> dis(-0.01, 0.01);

//     double distX = dis(_randomGenerator);
//     double distY = dis(_randomGenerator);

//     locationX += distX;
//     locationY += distY;
//   }

//   printf("[Korali] Initial Conditions:\n");
//   printf("[Korali] locationX: %f\n", locationX);
//   printf("[Korali] locationY: %f\n", locationY);

//   // Setting initial position and orientation for the fish
//   double C[2] = { locationX, locationY};
//   agent->setCenterOfMass(C);

//   // After moving the agent, the obstacles have to be restarted
//   _environment->startObstacles();
// }

// std::vector<double> logDivision(double start, double end, size_t nvertices)
// {
//     std::vector<double> vertices(nvertices, 0.0);
//     for(size_t idx = 0; idx < nvertices; ++idx)
//     {
//         vertices[idx] = std::exp((double) idx / (double) (nvertices-1.0) * std::log(end-start+1.0)) - 1.0 + start;
// 	printf("v %zu %lf\n", idx, vertices[idx]);
//     }
//     return vertices;
// }

//////////////////////////////////////////////////////// Other environments

/*
// Swimmer following an obstacle
void runEnvironmentCmaes(korali::Sample &s)
{
  // Setting seed
  size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Creating results directory
  std::string baseDir = "_results_transport_cmaes";
  char resDir[64];
  sprintf(resDir, "%s/sample%08lu", baseDir.c_str(), sampleId);
  std::filesystem::create_directories(resDir); 
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

  // Obtaining agent
  SmartCylinder* agent = dynamic_cast<SmartCylinder *>(_environment->getShapes()[0]);

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = 0.0;

  // Reseting environment and setting initial conditions
  _environment->resetRL();
  std::vector<double> start{0.2,0.5};
  setInitialConditions(agent, start, false);

  // Set target 
  std::vector<double> target{0.8,0.5};

  // Setting initial state
  auto state = agent->state( target );

  // Parametrization of forces
  std::vector<double> params = s["Parameters"];
  size_t numParams = params.size();

  double* centerArr = agent->center;
  std::vector<double> currentPos(centerArr, centerArr+2);
  double dist = distance(currentPos, target);
  double dDist = dist / (float) numParams;

  double t = 0.0;      // Current time
  size_t curStep = 0;  // current Step
  double distanceNextAct = dist;

  // Setting maximum number of steps before truncation
  size_t maxSteps = 1e6;
  double distThreshold = 1e-3;
  if(dDist <= distThreshold)
  {
    fprintf(stderr, "Decrease distance threshold bewlow %f due to large amount of params\n", dDist);
    exit(-1);
  }
 
  // Starting main environment loop
  bool done = false;
  size_t forceIdx = 0;
  double force = params[forceIdx];
  std::vector<double> action(2, 0.0);

  while (done == false && curStep < maxSteps)
  {
    centerArr = agent->center;
    currentPos[0] = centerArr[0];
    currentPos[1] = centerArr[1];

    if (dist < distanceNextAct)
    {
	distanceNextAct -= dDist;
	force = params[++forceIdx];
    }

    action[0] = force*(target[0]-currentPos[0])/dist;
    action[1] = force*(target[1]-currentPos[1])/dist;

    // Setting action
    agent->act( action );
    double dt = _environment->calcMaxTimestep();
    t += dt;
 
    bool error = _environment->advance(dt);
    dist = distance(currentPos, target);
    done = (dist <= distThreshold);
 
    curStep++;
    
    // Printing Information:
    printf("[Korali] Sample %lu, Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    printf("[Korali] State: [ %.6f, %.6f ]\n", currentPos[0], currentPos[1]);
    printf("[Korali] Force: [ %.6f, %.6f ]\n", action[0], action[1]);
    printf("[Korali] Energy %f, Distance %f, Terminal?: %d\n", agent->energy, dist, done);
    printf("[Korali] Time: %.3fs\n", t);
    printf("[Korali] -------------------------------------------------------\n");
    fflush(stdout);
 
    if (error == true)
    {
      fprintf(stderr, "Error during environment\n");
      exit(-1);
    }
 

  }
  
  if(forceIdx != numParams)
  {
      fprintf(stderr, "Error during sanity check, forceIdx %zu, expected %zu\n", forceIdx, numParams);
  }  

  // Reward is square deviation from 2sec (dummy)
  s["F(x)"] = -1.0*(t-2.0)*(t-2.0)-dist;

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

  // Closing log file
  fclose(logFile);
}

*/

/*
// Swimmer following an obstacle
void optimizeTimeWithEnergyBudget(korali::Sample &s)
{
  size_t sampleId = s["Sample Id"];
  double startX = 0.2;
  double endX = 0.8;
  double energyBudget = 1.0;

  // Creating results directory
  std::string baseDir = "_results_transport_cmaes";
  char resDir[64];
  sprintf(resDir, "%s/sample%08lu", baseDir.c_str(), sampleId);
  std::filesystem::create_directories(resDir); 
  
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

  // Obtaining agent
  SmartCylinder* agent = dynamic_cast<SmartCylinder *>(_environment->getShapes()[0]);

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = 0.0;

  // Reseting environment and setting initial conditions
  _environment->resetRL();
  std::vector<double> start{startX,0.5};
  setInitialConditions(agent, start, false);

  // Set target 
  std::vector<double> target{endX,0.5};

  // Setting initial state
  auto state = agent->state( target );

  // Parametrization of forces
  std::vector<double> params = s["Parameters"];
  std::sort(params.begin(), params.end()); 
  size_t numParams = params.size();

  double* centerArr = agent->center;
  std::vector<double> currentPos(centerArr, centerArr+2);
  std::vector<double> vertices = logDivision(startX, endX, numParams+1);
  std::vector<double> energyBudgets(numParams+1);
  energyBudgets[0] = energyBudget * params[0];
  for(size_t idx = 1; idx < numParams; ++idx)
  {
  	energyBudgets[idx] = energyBudget * (params[idx]-params[idx-1]);
  }
  energyBudgets[numParams] = energyBudget * (1.0 - params[numParams-1]);

  double dist = distance(currentPos, target);
  double dDist = dist / (float) numParams;

  double t = 0.0;      // Current time
  size_t curStep = 0;  // current Step

  // Setting maximum number of steps before truncation
  size_t maxSteps = 1e6;
  double distThreshold = 1e-3;
  if(dDist <= distThreshold)
  {
    fprintf(stderr, "Decrease distance threshold bewlow %f due to large amount of params\n", dDist);
    exit(-1);
  }
 
  // Starting main environment loop
  bool done = false;
  size_t forceIdx = 0;
  double force = params[forceIdx++];
  std::vector<double> action(2, 0.0);

  while (done == false && curStep < maxSteps)
  {
    centerArr = agent->center;
    currentPos[0] = centerArr[0];
    currentPos[1] = centerArr[1];

    if (forceIdx+1 < vertices.size())
    if (currentPos[0] >= vertices[forceIdx+1])
    {
	forceIdx++;
	force = params[forceIdx];
    }

    action[0] = force*(target[0]-currentPos[0])/dist;
    action[1] = force*(target[1]-currentPos[1])/dist;

    // Setting action
    agent->act( action );
    double dt = _environment->calcMaxTimestep();
    t += dt;
 
    bool error = _environment->advance(dt);
    dist = distance(currentPos, target);
    done = (dist <= distThreshold);
 
    curStep++;
    
    // Printing Information:
    printf("[Korali] Sample %lu, Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    printf("[Korali] State: [ %.6f, %.6f ]\n", currentPos[0], currentPos[1]);
    printf("[Korali] Force: [ %.6f, %.6f ]\n", action[0], action[1]);
    printf("[Korali] Energy %f, Distance %f, Terminal?: %d\n", agent->energy, dist, done);
    printf("[Korali] Time: %.3fs\n", t);
    printf("[Korali] -------------------------------------------------------\n");
    fflush(stdout);
 
    if (error == true)
    {
      fprintf(stderr, "Error during environment\n");
      exit(-1);
    }
 

  }
  
  if(forceIdx != numParams)
  {
      fprintf(stderr, "Error during sanity check, forceIdx %zu, expected %zu\n", forceIdx, numParams);
  }  

  // Reward is square deviation from 2sec (dummy)
  s["F(x)"] = -1.0*(t-2.0)*(t-2.0)-dist;

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

  // Closing log file
  fclose(logFile);
}
*/

/*
void runEnvironmentMocmaes(korali::Sample &s)
{
  // Defining constants
  size_t sampleId = s["Sample Id"];
  double startX = 0.2;
  double endX = 0.8;
  double height = 0.5;

  // Constraints
  size_t maxSteps = 1e6;
  size_t maxEnergy = 1e-2;
 
  // Creating results directory
  std::string baseDir = "true_results_transport_mocmaes0/";
  char resDir[64];
  sprintf(resDir, "%s/sample%08lu", baseDir.c_str(), sampleId);
  std::filesystem::create_directories(resDir); 
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

  // Obtaining agent
  SmartCylinder* agent = dynamic_cast<SmartCylinder *>(_environment->getShapes()[0]);

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = 0.0;

  // Reseting environment and setting initial conditions
  _environment->resetRL();
  std::vector<double> start{startX, height};
  setInitialConditions(agent, start, false);

  // Set target 
  std::vector<double> target{endX, height};

  // Setting initial state
  auto state = agent->state( target );

  // Parametrization of force
  std::vector<double> params = s["Parameters"];
  size_t numParams = params.size();

  double* centerArr = agent->center;
  std::vector<double> currentPos(centerArr, centerArr+2);
  std::vector<double> vertices = logDivision(startX, endX, numParams+1);

  // Init counting variables
  double dist = distance(currentPos, target);
  double energy = 0.0; // Total energy
  double t = 0.0;      // Current time
  size_t curStep = 0;  // Current Step

  // Starting main environment loop
  bool done = false;
  size_t forceIdx = 0;
  double force = params[forceIdx];
  std::vector<double> action(2, 0.0);

  while (done == false)
  {
    centerArr = agent->center;
    currentPos[0] = centerArr[0];
    currentPos[1] = centerArr[1];

    if (forceIdx+1 < vertices.size())
    if (currentPos[0] >= vertices[forceIdx+1])
    {
	forceIdx++;
	force = params[forceIdx];
    }

    action[0] = force*(target[0]-currentPos[0])/dist;
    action[1] = force*(target[1]-currentPos[1])/dist;

    // Setting action
    agent->act( action );
    double dt = _environment->calcMaxTimestep();
    t += dt;
 
    // Update distance and check termination
    bool error = _environment->advance(dt);
    dist = distance(currentPos, target);
    energy = agent->energy;

    // Checkting termination
    done = (currentPos[0] >= endX) && (curStep <= maxSteps);
 
    curStep++;
    
    // Printing Information:
    printf("[Korali] Sample %lu, Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    printf("[Korali] State: [ %.6f, %.6f ]\n", currentPos[0], currentPos[1]);
    printf("[Korali] Force: [ %.6f, %.6f ]\n", action[0], action[1]);
    printf("[Korali] Energy %f, Distance %f, Terminal?: %d\n", energy, dist, done);
    printf("[Korali] Time: %.3fs\n", t);
    printf("[Korali] -------------------------------------------------------\n");
    fflush(stdout);
 
    if (error == true)
    {
      fprintf(stderr, "Error during environment\n");
      exit(-1);
    }

  }
  
  if(forceIdx != numParams)
  {
      fprintf(stderr, "Error during sanity check (sampleId %zu), forceIdx %zu, expected %zu\n", sampleId, forceIdx, numParams);
  }

  // Penalization for not reaching target
  if (currentPos[0] < endX)
  {
	t += (endX-currentPos[0])*1e9;
	energy += (endX-currentPos[0])*1e9;
  }
  if (energy > maxEnergy)
  {
	t += (energy-maxEnergy)*1e9;
	energy += (energy-maxEnergy)*1e9;
  }
  
  // Setting Objectives
  std::vector<double> objectives = { -t, -energy };
  printf("Objectives: %f (time), %f (energy)\n", t, energy);
  s["F(x)"] = objectives;

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

  // Closing log file
  fclose(logFile);

}
*/