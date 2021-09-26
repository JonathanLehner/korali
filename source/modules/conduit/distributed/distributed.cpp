#include "auxiliar/MPIUtils.hpp"
#include "engine.hpp"
#include "modules/conduit/distributed/distributed.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/solver.hpp"
#include "sample/sample.hpp"

using namespace std;

namespace korali
{
namespace conduit
{
;

void Distributed::initialize()
{
#ifndef _KORALI_USE_MPI
  KORALI_LOG_ERROR("Running an Distributed-based Korali application, but Korali was installed without support for MPI.\n");
#else

  // Sanity checks for the correct initialization of MPI
  int isInitialized = 0;
  MPI_Initialized(&isInitialized);
  if (isInitialized == 0) KORALI_LOG_ERROR("Korali requires that the MPI is initialized by the user (e.g., via MPI_init) prior to running the engine.\n");
  if (__isMPICommGiven == false) KORALI_LOG_ERROR("Korali requires that MPI communicator is passed (via setKoraliMPIComm) prior to running the engine.\n");
  if (_rankCount == 1) KORALI_LOG_ERROR("Korali Distributed applications require at least 2 Distributed ranks to run.\n");

  // Getting size and id from the user-defined communicator
  MPI_Comm_size(__KoraliGlobalMPIComm, &_rankCount);
  MPI_Comm_rank(__KoraliGlobalMPIComm, &_rankId);
  MPI_Barrier(__KoraliGlobalMPIComm);

  // Determining ranks per worker
  int curWorker = 0;
  _workerCount = (_rankCount - 1) / _ranksPerWorker;
  size_t workerRemainder = (_rankCount - 1) % _ranksPerWorker;
  if (workerRemainder != 0) KORALI_LOG_ERROR("Korali was instantiated with %lu MPI ranks (minus one for the engine), divided into %lu workers. This setup does not provide a perfectly divisible distribution, and %lu unused ranks remain.\n", _workerCount, _ranksPerWorker, workerRemainder);

  _localRankId = 0;
  _workerIdSet = false;

  // Storage to map MPI ranks to their corresponding worker
  _workerTeams.resize(_workerCount);
  for (int i = 0; i < _workerCount; i++)
    _workerTeams[i].resize(_ranksPerWorker);

  // Storage to map workers to MPI ranks
  _rankToWorkerMap.resize(_rankCount);

  // Initializing available worker queue
  _workerQueue = queue<size_t>();
  while (!_workerQueue.empty()) _workerQueue.pop();

  // Putting workers in the queue
  for (int i = 0; i < _workerCount; i++)
    _workerQueue.push(i);

  // Now assigning ranks to workers and viceversa
  int currentRank = 0;
  for (int i = 0; i < _workerCount; i++)
    for (int j = 0; j < _ranksPerWorker; j++)
    {
      if (currentRank == _rankId)
      {
        curWorker = i;
        _localRankId = j;
        _workerIdSet = true;
      }

      _workerTeams[i][j] = currentRank;
      _rankToWorkerMap[currentRank] = i;
      currentRank++;
    }

  // If this is the root rank, check whether the number of ranks is correct
  if (isRoot())
  {
    int mpiSize;
    MPI_Comm_size(__KoraliGlobalMPIComm, &mpiSize);

    checkRankCount();

    curWorker = _workerCount + 1;
  }

  // Creating communicator
  MPI_Comm_split(__KoraliGlobalMPIComm, curWorker, _rankId, &__koraliWorkerMPIComm);

  // Waiting for all ranks to reach this point
  MPI_Barrier(__KoraliGlobalMPIComm);
#endif
}

void Distributed::checkRankCount()
{
  if (_rankCount < _ranksPerWorker + 1)
    KORALI_LOG_ERROR("You are running Korali with %d ranks. However, you need at least %d ranks to have at least one worker team. \n", _rankCount, _ranksPerWorker + 1);
}

void Distributed::initServer()
{
#ifdef _KORALI_USE_MPI
  if (isRoot() == false && _workerIdSet == true) worker();
#endif
}

void Distributed::terminateServer()
{
#ifdef _KORALI_USE_MPI
  auto terminationJs = knlohmann::json();
  terminationJs["Conduit Action"] = "Terminate";

  string terminationString = terminationJs.dump();
  size_t terminationStringSize = terminationString.size();

  if (isRoot())
  {
    for (int i = 0; i < _workerCount; i++)
      for (int j = 0; j < _ranksPerWorker; j++)
        MPI_Send(terminationString.c_str(), terminationStringSize, MPI_CHAR, _workerTeams[i][j], __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm);
  }

#endif

  Conduit::finalize();
}

void Distributed::broadcastMessageToWorkers(knlohmann::json &message)
{
#ifdef _KORALI_USE_MPI
  // Run broadcast only if this is the master process
  if (!isRoot()) return;

  string messageString = message.dump();
  size_t messageStringSize = messageString.size();

  for (int i = 0; i < _workerCount; i++)
    for (int j = 0; j < _ranksPerWorker; j++)
      MPI_Send(messageString.c_str(), messageStringSize, MPI_CHAR, _workerTeams[i][j], __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm);
#endif
}

int Distributed::getRootRank()
{
#ifdef _KORALI_USE_MPI
  return _rankCount - 1;
#endif

  return 0;
}

bool Distributed::isRoot()
{
#ifdef _KORALI_USE_MPI
  return _rankId == getRootRank();
#endif

  return true;
}

bool Distributed::isWorkerLeadRank()
{
  // Arbitrarily, we decide that rank 0 is the root rank
  return _localRankId == 0;
}

void Distributed::sendMessageToEngine(knlohmann::json &message)
{
#ifdef _KORALI_USE_MPI
  if (_localRankId == 0)
  {
    string messageString = message.dump();
    size_t messageStringSize = messageString.size();
    MPI_Send(messageString.c_str(), messageStringSize, MPI_CHAR, getRootRank(), __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm);
  }
#endif
}

knlohmann::json Distributed::recvMessageFromEngine()
{
  auto message = knlohmann::json();

#ifdef _KORALI_USE_MPI
  MPI_Barrier(__koraliWorkerMPIComm);

  MPI_Status status;
  MPI_Probe(getRootRank(), __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm, &status);
  int messageSize = 0;
  MPI_Get_count(&status, MPI_CHAR, &messageSize);

  char *jsonStringChar = (char *)malloc(sizeof(char) * (messageSize + 1));
  MPI_Recv(jsonStringChar, messageSize, MPI_CHAR, getRootRank(), __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm, MPI_STATUS_IGNORE);

  jsonStringChar[messageSize] = '\0';
  message = knlohmann::json::parse(jsonStringChar);
  free(jsonStringChar);
#endif

  return message;
}

void Distributed::listenWorkers()
{
#ifdef _KORALI_USE_MPI

  // Scanning all incoming messages
  int foundMessage = 0;

  // Reading pending messages from any worker
  MPI_Status status;
  MPI_Iprobe(MPI_ANY_SOURCE, __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm, &foundMessage, &status);

  // If message found, receive it and storing in the corresponding sample's queue
  if (foundMessage == 1)
  {
    // Obtaining source rank, worker ID, and destination sample from the message
    int source = status.MPI_SOURCE;
    int worker = _rankToWorkerMap[source];
    auto sample = _workerToSampleMap[worker];

    // Receiving message from the worker
    int messageSize = 0;
    MPI_Get_count(&status, MPI_CHAR, &messageSize);
    char *messageStringChar = (char *)malloc(sizeof(char) * (messageSize + 1));
    MPI_Recv(messageStringChar, messageSize, MPI_CHAR, source, __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm, MPI_STATUS_IGNORE);
    messageStringChar[messageSize] = '\0';
    auto message = knlohmann::json::parse(messageStringChar);
    free(messageStringChar);

    // Storing message in the sample message queue
    sample->_messageQueue.push(message);
  }

#endif
}

void Distributed::sendMessageToSample(Sample &sample, knlohmann::json &message)
{
#ifdef _KORALI_USE_MPI
  string messageString = message.dump();
  size_t messageStringSize = messageString.size();

  for (int i = 0; i < _ranksPerWorker; i++)
  {
    int rankId = _workerTeams[sample._workerId][i];
    MPI_Send(messageString.c_str(), messageStringSize, MPI_CHAR, rankId, __KORALI_MPI_MESSAGE_JSON_TAG, __KoraliGlobalMPIComm);
  }
#endif
}

void Distributed::stackEngine(Engine *engine)
{
#ifdef _KORALI_USE_MPI
  // (Engine-Side) Adding engine to the stack to support Korali-in-Korali execution
  _engineStack.push(engine);

  knlohmann::json engineJs;
  engineJs["Conduit Action"] = "Stack Engine";
  engine->serialize(engineJs["Engine"]);

  broadcastMessageToWorkers(engineJs);
#endif
}

void Distributed::popEngine()
{
#ifdef _KORALI_USE_MPI
  // (Engine-Side) Removing the current engine to the conduit's engine stack
  _engineStack.pop();

  auto popJs = knlohmann::json();
  popJs["Conduit Action"] = "Pop Engine";
  broadcastMessageToWorkers(popJs);
#endif
}

size_t Distributed::getProcessId()
{
  return _rankId;
}

void Distributed::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Ranks Per Worker"))
 {
 try { _ranksPerWorker = js["Ranks Per Worker"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ distributed ] \n + Key:    ['Ranks Per Worker']\n%s", e.what()); } 
   eraseValue(js, "Ranks Per Worker");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Ranks Per Worker'] required by distributed.\n"); 

 Conduit::setConfiguration(js);
 _type = "distributed";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: distributed: \n%s\n", js.dump(2).c_str());
} 

void Distributed::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Ranks Per Worker"] = _ranksPerWorker;
 Conduit::getConfiguration(js);
} 

void Distributed::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Ranks Per Worker\": 1}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Conduit::applyModuleDefaults(js);
} 

void Distributed::applyVariableDefaults() 
{

 Conduit::applyVariableDefaults();
} 

;

} //conduit
} //korali
;
