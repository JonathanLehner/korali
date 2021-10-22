/** \namespace solver
* @brief Namespace declaration for modules of type: solver.
*/

/** \file
* @brief Header file for module: Learner.
*/

/** \dir solver/learner
* @brief Contains code, documentation, and scripts for module: Learner.
*/

#pragma once

#include "modules/solver/solver.hpp"

namespace korali
{
namespace solver
{
;

/**
* @brief Class declaration for module: Learner.
*/
class Learner : public Solver
{
  public: 
  
 
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
   * @brief For learner modules which have been trained, test returns an inferred output batch, from a batch of inputs to process.
   * @param input The inputs from which to infer outputs. Format: BxTxIC (B: Batch Size, T: Time steps, IC: Input channels)
   * @return The inferred batch outputs for the last given timestep. Format: BxOC (B: Batch Size, OC: Output channels)
   */
  virtual std::vector<std::vector<float>> &getEvaluation(const std::vector<std::vector<std::vector<float>>> &input);

  /**
   * @brief Returns the hyperparameters required to continue training in the future
   * @return The hyperparameters
   */
  virtual std::vector<float> getHyperparameters() = 0;

  /**
   * @brief Sets the hyperparameters required to continue training from a previous state
   * @param hyperparameters The hyperparameters to use
   */
  virtual void setHyperparameters(const std::vector<float> &hyperparameters) = 0;
};

} //solver
} //korali
;
