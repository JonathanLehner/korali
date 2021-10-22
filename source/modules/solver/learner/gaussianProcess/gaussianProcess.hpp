/** \namespace learner
* @brief Namespace declaration for modules of type: learner.
*/

/** \file
* @brief Header file for module: GaussianProcess.
*/

/** \dir solver/learner/gaussianProcess
* @brief Contains code, documentation, and scripts for module: GaussianProcess.
*/

#pragma once

#include "engine.hpp"

#include "auxiliar/libgp/gp.h"

#include "modules/experiment/experiment.hpp"
#include "modules/problem/supervisedLearning/supervisedLearning.hpp"
#include "modules/solver/learner/learner.hpp"

#include <memory>

namespace korali
{
namespace solver
{
namespace learner
{
;

/**
* @brief Class declaration for module: GaussianProcess.
*/
class GaussianProcess : public Learner
{
  public: 
  /**
  * @brief Covariance function for the libgp library.
  */
   std::string _covarianceFunction;
  /**
  * @brief Default value of the hyperparameters, used to initialize the Gaussian Processes.
  */
   float _defaultHyperparameter;
  /**
  * @brief Represents the state and configuration of the optimization algorithm.
  */
   knlohmann::json _optimizer;
  /**
  * @brief [Internal Use] Dimension of the input space.
  */
   size_t _gpInputDimension;
  /**
  * @brief [Internal Use] Number of the Gaussian Process' parameters.
  */
   size_t _gpParameterDimension;
  /**
  * @brief [Internal Use] Gaussian Process' hyperparameters.
  */
   std::vector<float> _gpHyperparameters;
  /**
  * @brief [Termination Criteria] Execution will end as soon as the internal optimizer reaches one of its termination criteria.
  */
   int _terminateWithOptimizer;
  
 
  /**
  * @brief Determines whether the module can trigger termination of an experiment run.
  * @return True, if it should trigger termination; false, otherwise.
  */
  bool checkTermination() override;
  /**
  * @brief Obtains the entire current state and configuration of the module.
  * @param js JSON object onto which to save the serialized state of the module.
  */
  void getConfiguration(knlohmann::json& js) override;
  /**
  * @brief Sets the entire state and configuration of the module, given a JSON object.
  * @param js JSON object from which to deserialize the state of the module.
  */
  void setConfiguration(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default configuration upon its creation.
  * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  void applyModuleDefaults(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  void applyVariableDefaults() override;
  

  /**
   * @brief Storage for the output values
   */
  std::vector<std::vector<float>> _outputValues;

  /**
   * @brief Korali engine for optimizing NN weights and biases
   */
  problem::SupervisedLearning *_problem;

  /**
   * @brief Pointer to the gaussian processes library
   */
  std::unique_ptr<libgp::GaussianProcess> _gp;

  /**
   * @brief Korali experiment for optimizing the GP's parameters
   */
  Experiment _koraliExperiment;

  std::vector<std::vector<float>> &getEvaluation(const std::vector<std::vector<std::vector<float>>> &input) override;
  std::vector<float> getHyperparameters() override;
  void setHyperparameters(const std::vector<float> &hyperparameters) override;
  void initialize() override;
  void runGeneration() override;
  void printGenerationAfter() override;
};

} //learner
} //solver
} //korali
;
