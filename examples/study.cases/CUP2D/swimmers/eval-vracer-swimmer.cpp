// Select which environment to use
#include "_model/swimmerEnvironment.hpp"
#include "korali.hpp"

int main(int argc, char *argv[])
{
  // Gathering actual arguments from MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided != MPI_THREAD_FUNNELED)
  {
    printf("Error initializing MPI\n");
    exit(-1);
  }

  // Storing parameters
  _argc = argc;
  _argv = argv;

  // Getting number of workers
  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine

  // Setting results path
  std::string trainingResultsPath = "_trainingResults/";
  std::string testingResultsPath = "_testingResults/";

  // Creating Korali experiment
  auto e = korali::Experiment();

  // Check if there is log files to continue training
  auto found = e.loadState(trainingResultsPath+"/latest");
  if (found == true) printf("[Korali] Evaluation results found...\n");
  else { fprintf(stderr, "[Korali] Error: cannot find previous results\n"); exit(0); } 

  // Creating Korali engine
  auto k = korali::Engine();

  // Set conduit and pass MPI communicator
  k["Conduit"]["Type"] = "Distributed";
  korali::setKoraliMPIComm(MPI_COMM_WORLD);

  // Adding custom setting to run the environment dumping the state files during testing
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.1;
  e["Problem"]["Custom Settings"]["Dump Path"] = testingResultsPath;
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["File Output"]["Path"] = trainingResultsPath;
  e["Solver"]["Mode"] = "Testing";
  // e["Solver"]["Testing"]["Best Policy"] = e["Solver"]["Training"]["Best Policy"];
  // e["Solver"]["Training"]["Current Policy"] = donor["Solver"]["Training"]["Best Policy"];
  for (int i = 0; i < N; i++) e["Solver"]["Testing"]["Sample Ids"][i] = i;

  k.run(e);
}
