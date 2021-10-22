#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/solver/learner/gaussianProcess/gaussianProcess.hpp"
#include "sample/sample.hpp"

#include <Eigen/Dense>

namespace korali
{
namespace solver
{
namespace learner
{
;

/**
 * @brief Converts a vector of floats to Eigen format
 * @param v the vector to convert
 * @return An Eigen vector type
 */
Eigen::VectorXd
toEigen(const std::vector<float> &v)
{
  Eigen::VectorXd ev(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    ev[i] = v[i];
  return ev;
}

/**
 * @brief Model function to evaluate the error function of the GP
 * @param sample The sample containing the proposal parameters
 * @param gp Pointer to the GP
 */
void runSample(Sample &sample, libgp::GaussianProcess *gp)
{
  size_t gpParameterDimension = gp->covf().get_param_dim();
  const Eigen::VectorXd p = toEigen(sample["Parameters"].get<std::vector<float>>());

  gp->covf().set_loghyper(p);

  sample["F(x)"] = gp->log_likelihood();
  sample["logP(x)"] = sample["F(x)"];

  Eigen::VectorXd eigenGrad = gp->log_likelihood_gradient();
  for (size_t i = 0; i < gpParameterDimension; i++)
    sample["Gradient"][i] = eigenGrad[i];
}

void GaussianProcess::initialize()
{
  _problem = dynamic_cast<problem::SupervisedLearning *>(_k->_problem);

  if (_problem->_maxTimesteps > 1) KORALI_LOG_ERROR("Training data cannot be time-dependent.");
  if (_problem->_trainingBatchSize == 0) KORALI_LOG_ERROR("Training data has not been provided for variable 0.");
  if (_problem->_solutionSize > 1) KORALI_LOG_ERROR("The solution space should be one dimensional.");

  // Checking that incoming data has a correct format
  _problem->verifyData();

  _gpInputDimension = _problem->_inputSize;
  _gp = std::make_unique<libgp::GaussianProcess>(_gpInputDimension, _covarianceFunction);

  _gpParameterDimension = _gp->covf().get_param_dim();

  // Creating evaluation lambda function for optimization
  auto evaluateProposal = [gp = _gp.get()](Sample &sample)
  {
    runSample(sample, gp);
  };

  _koraliExperiment["Problem"]["Type"] = "Optimization";
  _koraliExperiment["Problem"]["Objective Function"] = evaluateProposal;

  Eigen::VectorXd eParameters(_gpParameterDimension);

  for (size_t i = 0; i < _gpParameterDimension; i++)
  {
    _koraliExperiment["Variables"][i]["Name"] = "X" + std::to_string(i);
    eParameters[i] = _defaultHyperparameter;
    _koraliExperiment["Variables"][i]["Initial Value"] = eParameters[i];
  }
  _gp->covf().set_loghyper(eParameters);

  _koraliExperiment["Solver"] = _optimizer;
  _koraliExperiment["Solver"]["Termination Criteria"]["Max Generations"] = 1;

  _koraliExperiment["File Output"]["Frequency"] = 0;
  _koraliExperiment["File Output"]["Enabled"] = false;
  _koraliExperiment["Console Output"]["Frequency"] = 0;
  _koraliExperiment["Console Output"]["Verbosity"] = "Silent";
  _koraliExperiment["Random Seed"] = _k->_randomSeed++;

  // Pass the training data from korali to the GP library
  double inData[_gpInputDimension];
  double outData;

  // Running initialization to verify that the configuration is correct
  _koraliExperiment.initialize();

  for (size_t i = 0; i < _problem->_trainingBatchSize; i++)
  {
    for (size_t j = 0; j < _gpInputDimension; j++)
      inData[j] = _problem->_inputData[i][0][j];

    outData = _problem->_solutionData[i][0];
    _gp->add_pattern(inData, outData);
  }
}

void GaussianProcess::runGeneration()
{
  _koraliExperiment["Solver"]["Termination Criteria"]["Max Generations"] = _koraliExperiment._currentGeneration + 1;
  korali::Engine engine;
  engine.run(_koraliExperiment);
  _gpHyperparameters = _koraliExperiment["Results"]["Best Sample"]["Parameters"].get<std::vector<float>>();
}

void GaussianProcess::printGenerationAfter()
{
  return;
}

std::vector<std::vector<float>> &GaussianProcess::getEvaluation(const std::vector<std::vector<std::vector<float>>> &input)
{
  _outputValues.resize(1);
  _outputValues[0].resize(2);

  if (input.size() > 1) KORALI_LOG_ERROR("Gaussian Process does not support multi-timestep evaluation.\n");
  if (input[0].size() > 1) KORALI_LOG_ERROR("Gaussian Process does not support minibatch evaluation.\n");

  _gp->covf().set_loghyper(toEigen(_gpHyperparameters));

  std::vector<double> inputData(input[0][0].begin(), input[0][0].end());

  _outputValues[0][0] = _gp->f(inputData.data());
  _outputValues[0][1] = _gp->var(inputData.data());

  return _outputValues;
}

std::vector<float> GaussianProcess::getHyperparameters()
{
  return _gpHyperparameters;
}

void GaussianProcess::setHyperparameters(const std::vector<float> &hyperparameters)
{
  _gpHyperparameters = hyperparameters;
}

void GaussianProcess::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "gp Input Dimension"))
 {
 try { _gpInputDimension = js["gp Input Dimension"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ gaussianProcess ] \n + Key:    ['gp Input Dimension']\n%s", e.what()); } 
   eraseValue(js, "gp Input Dimension");
 }

 if (isDefined(js, "gp Parameter Dimension"))
 {
 try { _gpParameterDimension = js["gp Parameter Dimension"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ gaussianProcess ] \n + Key:    ['gp Parameter Dimension']\n%s", e.what()); } 
   eraseValue(js, "gp Parameter Dimension");
 }

 if (isDefined(js, "gp Hyperparameters"))
 {
 try { _gpHyperparameters = js["gp Hyperparameters"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ gaussianProcess ] \n + Key:    ['gp Hyperparameters']\n%s", e.what()); } 
   eraseValue(js, "gp Hyperparameters");
 }

 if (isDefined(js, "Covariance Function"))
 {
 try { _covarianceFunction = js["Covariance Function"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ gaussianProcess ] \n + Key:    ['Covariance Function']\n%s", e.what()); } 
   eraseValue(js, "Covariance Function");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Covariance Function'] required by gaussianProcess.\n"); 

 if (isDefined(js, "Default Hyperparameter"))
 {
 try { _defaultHyperparameter = js["Default Hyperparameter"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ gaussianProcess ] \n + Key:    ['Default Hyperparameter']\n%s", e.what()); } 
   eraseValue(js, "Default Hyperparameter");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Default Hyperparameter'] required by gaussianProcess.\n"); 

 if (isDefined(js, "Optimizer"))
 {
 _optimizer = js["Optimizer"].get<knlohmann::json>();

   eraseValue(js, "Optimizer");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Optimizer'] required by gaussianProcess.\n"); 

 if (isDefined(js, "Termination Criteria", "Terminate With Optimizer"))
 {
 try { _terminateWithOptimizer = js["Termination Criteria"]["Terminate With Optimizer"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ gaussianProcess ] \n + Key:    ['Termination Criteria']['Terminate With Optimizer']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Terminate With Optimizer");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Terminate With Optimizer'] required by gaussianProcess.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Learner::setConfiguration(js);
 _type = "learner/gaussianProcess";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: gaussianProcess: \n%s\n", js.dump(2).c_str());
} 

void GaussianProcess::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Covariance Function"] = _covarianceFunction;
   js["Default Hyperparameter"] = _defaultHyperparameter;
   js["Optimizer"] = _optimizer;
   js["Termination Criteria"]["Terminate With Optimizer"] = _terminateWithOptimizer;
   js["gp Input Dimension"] = _gpInputDimension;
   js["gp Parameter Dimension"] = _gpParameterDimension;
   js["gp Hyperparameters"] = _gpHyperparameters;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Learner::getConfiguration(js);
} 

void GaussianProcess::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Default Hyperparameter\": 0.1, \"Termination Criteria\": {\"Terminate With Optimizer\": true}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Learner::applyModuleDefaults(js);
} 

void GaussianProcess::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Learner::applyVariableDefaults();
} 

bool GaussianProcess::checkTermination()
{
 bool hasFinished = false;

 if (_terminateWithOptimizer && _koraliExperiment._solver->checkTermination())
 {
  _terminationCriteria.push_back("gaussianProcess['Terminate With Optimizer'] = " + std::to_string(_terminateWithOptimizer) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Learner::checkTermination();
 return hasFinished;
}

;

} //learner
} //solver
} //korali
;
