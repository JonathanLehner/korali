#include "engine.hpp"
#include "modules/solver/agent/discrete/dVRACER/dVRACER.hpp"
#include "omp.h"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace agent
{
namespace discrete
{
;

void dVRACER::initializeAgent()
{
  // Initializing common discrete agent configuration
  Discrete::initializeAgent();

  /*********************************************************************
   * Initializing Critic/Policy Neural Network Optimization Experiment
   *********************************************************************/

  _criticPolicyExperiment["Problem"]["Type"] = "Supervised Learning";
  _criticPolicyExperiment["Problem"]["Max Timesteps"] = _timeSequenceLength;
  _criticPolicyExperiment["Problem"]["Training Batch Size"] = _miniBatchSize;
  _criticPolicyExperiment["Problem"]["Inference Batch Size"] = 1;
  _criticPolicyExperiment["Problem"]["Input"]["Size"] = _problem->_stateVectorSize;
  _criticPolicyExperiment["Problem"]["Solution"]["Size"] = 1 + _policyParameterCount; // The value function, action q values, and inverse temperature

  _criticPolicyExperiment["Solver"]["Type"] = "Learner/DeepSupervisor";
  _criticPolicyExperiment["Solver"]["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
  _criticPolicyExperiment["Solver"]["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
  _criticPolicyExperiment["Solver"]["Learning Rate"] = _currentLearningRate;
  _criticPolicyExperiment["Solver"]["Loss Function"] = "Direct Gradient";
  _criticPolicyExperiment["Solver"]["Steps Per Generation"] = 1;
  _criticPolicyExperiment["Solver"]["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
  _criticPolicyExperiment["Solver"]["Neural Network"]["Engine"] = _neuralNetworkEngine;
  _criticPolicyExperiment["Solver"]["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
  _criticPolicyExperiment["Solver"]["Output Weights Scaling"] = 0.001;

  // No transformations for the state value output
  _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Scale"][0] = 1.0f;
  _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Shift"][0] = 0.0f;
  _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][0] = "Identity";

  // No transofrmation for the q values
  for (size_t i = 0; i < _problem->_possibleActions.size(); ++i)
  {
    _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Scale"][i + 1] = 1.0f;
    _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Shift"][i + 1] = 0.0f;
    _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][i + 1] = "Identity";
  }

  // Transofrmation for the inverse temperature
  _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Scale"][1 + _problem->_possibleActions.size()] = 1.0f;
  _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Shift"][1 + _problem->_possibleActions.size()] = 0.0f;
  _criticPolicyExperiment["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][1 + _problem->_possibleActions.size()] = "Sigmoid";

  // Running initialization to verify that the configuration is correct
  _criticPolicyExperiment.initialize();
  _criticPolicyProblem = dynamic_cast<problem::SupervisedLearning *>(_criticPolicyExperiment._problem);
  _criticPolicyLearner = dynamic_cast<solver::learner::DeepSupervisor *>(_criticPolicyExperiment._solver);
}

void dVRACER::trainPolicy()
{
  // Obtaining Minibatch experience ids
  const auto miniBatch = generateMiniBatch(_miniBatchSize);

  // Gathering state sequences for selected minibatch
  const auto stateSequence = getMiniBatchStateSequence(miniBatch);

  // Running policy NN on the Minibatch experiences
  const auto policyInfo = runPolicy(stateSequence);

  // Using policy information to update experience's metadata
  updateExperienceMetadata(miniBatch, policyInfo);

  // Now calculating policy gradients
  calculatePolicyGradients(miniBatch);

  // Updating learning rate for critic/policy learner guided by REFER
  _criticPolicyLearner->_learningRate = _currentLearningRate;

  // Now applying gradients to update policy NN
  _criticPolicyLearner->runGeneration();
}

void dVRACER::calculatePolicyGradients(const std::vector<size_t> &miniBatch)
{
  const size_t miniBatchSize = miniBatch.size();

#pragma omp parallel for
  for (size_t b = 0; b < miniBatchSize; b++)
  {
    // Getting index of current experiment
    size_t expId = miniBatch[b];

    // Getting experience policy data
    const auto &expPolicy = _expPolicyVector[expId];

    // Getting current policy data
    const auto &curPolicy = _curPolicyVector[expId];

    // Getting value evaluation
    const float V = _stateValueVector[expId];
    const float expVtbc = _retraceValueVector[expId];

    // Storage for the update gradient
    std::vector<float> gradientLoss(1 + _policyParameterCount);

    // Gradient of Value Function V(s) (eq. (9); *-1 because the optimizer is maximizing)
    gradientLoss[0] = expVtbc - V;

    // Compute policy gradient only if inside trust region (or offPolicy disabled)
    if (_isOnPolicyVector[expId])
    {
      // Qret for terminal state is just reward
      float Qret = getScaledReward(_environmentIdVector[expId], _rewardVector[expId]);

      // If experience is non-terminal, add Vtbc
      if (_terminationVector[expId] == e_nonTerminal)
      {
        const float nextExpVtbc = _retraceValueVector[expId + 1];
        Qret += _discountFactor * nextExpVtbc;
      }

      // If experience is truncated, add truncated state value
      if (_terminationVector[expId] == e_truncated)
      {
        const float nextExpVtbc = _truncatedStateValueVector[expId];
        Qret += _discountFactor * nextExpVtbc;
      }

      // Compute Off-Policy Objective (eq. 5)
      const float lossOffPolicy = Qret - V;

      // Compute Policy Gradient wrt Params
      auto polGrad = calculateImportanceWeightGradient(curPolicy, expPolicy);

      // Set Gradient of Loss wrt Params
      for (size_t i = 0; i < _policyParameterCount; i++)
      {
        // '-' because the optimizer is maximizing
        gradientLoss[1 + i] = _experienceReplayOffPolicyREFERBeta * lossOffPolicy * polGrad[i];
      }
    }

    // Compute derivative of kullback-leibler divergence wrt current distribution params
    auto klGrad = calculateKLDivergenceGradient(expPolicy, curPolicy);

    // Step towards old policy (gradient pointing to larger difference between old and current policy)
    const float klGradMultiplier = -(1.0f - _experienceReplayOffPolicyREFERBeta);
    for (size_t i = 0; i < _policyParameterCount; i++)
      gradientLoss[1 + i] += klGradMultiplier * klGrad[i];

    // Set Gradient of Loss as Solution
    _criticPolicyProblem->_solutionData[b] = gradientLoss;
  }
}

std::vector<policy_t> dVRACER::runPolicy(const std::vector<std::vector<std::vector<float>>> &stateBatch)
{
  // Getting batch size
  size_t batchSize = stateBatch.size();

  // Storage for policy
  std::vector<policy_t> policyVector(batchSize);

  // Forward the neural network for this state
  const auto evaluation = _criticPolicyLearner->getEvaluation(stateBatch);

#pragma omp parallel for
  for (size_t b = 0; b < batchSize; b++)
  {
    // Getting state value
    policyVector[b].stateValue = evaluation[b][0];

    // Storage for action probabilities
    float maxq = -korali::Inf;
    std::vector<float> pActions(_problem->_possibleActions.size());
    std::vector<float> qValAndInvTemp(_policyParameterCount);

    // Get the inverse of the temperature for the softmax distribution
    const float invTemperature = evaluation[b][_policyParameterCount];

    // Iterating all Q(s,a)
    for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
    {
      // Computing Q(s,a_i)
      qValAndInvTemp[i] = evaluation[b][1 + i];

      // Extracting max Q(s,a_i)
      if (qValAndInvTemp[i] > maxq) maxq = qValAndInvTemp[i];
    }

    // Storage for the cumulative e^Q(s,a_i)/maxq
    float sumExpQVal = 0.0;

    for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
    {
      // Computing e^(Q(s,a_i) - maxq)
      const float expCurQVal = std::exp(invTemperature * (qValAndInvTemp[i] - maxq));

      // Computing Sum_i(e^Q(s,a_i)/e^maxq)
      sumExpQVal += expCurQVal;

      // Storing partial value of the probability of the action
      pActions[i] = expCurQVal;
    }

    // Calculating inverse of Sum_i(e^Q(s,a_i))
    const float invSumExpQVal = 1.0f / sumExpQVal;

    // Normalizing action probabilities
    for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
      pActions[i] *= invSumExpQVal;

    // Set inverse temperature parameter
    qValAndInvTemp[_problem->_possibleActions.size()] = invTemperature;

    // Storing the action probabilities, the qVals and the inverse temperature into the policyy
    policyVector[b].actionProbabilities = pActions;
    policyVector[b].distributionParameters = qValAndInvTemp;
  }

  return policyVector;
}

knlohmann::json dVRACER::getAgentPolicy()
{
  knlohmann::json hyperparameters;
  hyperparameters["Policy"] = _criticPolicyLearner->getHyperparameters();
  return hyperparameters;
}

void dVRACER::setAgentPolicy(const knlohmann::json &hyperparameters)
{
  _criticPolicyLearner->setHyperparameters(hyperparameters["Policy"].get<std::vector<float>>());
}

void dVRACER::printAgentInformation()
{
  _k->_logger->logInfo("Normal", " + [dVRACER] Policy Learning Rate: %.3e\n", _currentLearningRate);
}

void dVRACER::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Statistics", "Average Action Sigmas"))
 {
 try { _statisticsAverageActionSigmas = js["Statistics"]["Average Action Sigmas"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ dVRACER ] \n + Key:    ['Statistics']['Average Action Sigmas']\n%s", e.what()); } 
   eraseValue(js, "Statistics", "Average Action Sigmas");
 }

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Initial Exploration Noise"))
 {
 try { _k->_variables[i]->_initialExplorationNoise = _k->_js["Variables"][i]["Initial Exploration Noise"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ dVRACER ] \n + Key:    ['Initial Exploration Noise']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Exploration Noise");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Exploration Noise'] required by dVRACER.\n"); 

 } 
 Discrete::setConfiguration(js);
 _type = "agent/discrete/dVRACER";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: dVRACER: \n%s\n", js.dump(2).c_str());
} 

void dVRACER::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Statistics"]["Average Action Sigmas"] = _statisticsAverageActionSigmas;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Initial Exploration Noise"] = _k->_variables[i]->_initialExplorationNoise;
 } 
 Discrete::getConfiguration(js);
} 

void dVRACER::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Discrete::applyModuleDefaults(js);
} 

void dVRACER::applyVariableDefaults() 
{

 std::string defaultString = "{\"Initial Exploration Noise\": -1.0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Discrete::applyVariableDefaults();
} 

bool dVRACER::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || Discrete::checkTermination();
 return hasFinished;
}

;

} //discrete
} //agent
} //solver
} //korali
;
