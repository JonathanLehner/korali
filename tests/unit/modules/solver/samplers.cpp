#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/sampler/sampler.hpp"
#include "modules/problem/sampling/sampling.hpp"
#include "modules/problem/bayesian/reference/reference.hpp"
#include "modules/solver/sampler/Nested/Nested.hpp"
#include "modules/solver/sampler/HMC/HMC.hpp"
#include "modules/solver/sampler/MCMC/MCMC.hpp"
#include "modules/solver/sampler/TMCMC/TMCMC.hpp"

namespace
{
 using namespace korali;
 using namespace korali::solver;
 using namespace korali::solver::sampler;
 using namespace korali::problem;
 using namespace korali::problem::bayesian;

 //////////////// BASE CLASS ////////////////////////

  TEST(samplers, baseClass)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();

   // Creating initial variable
   Variable v;
   e._variables.push_back(&v);
   e["Variables"][0]["Name"] = "Var 1";
   e["Variables"][0]["Initial Mean"] = 0.0;
   e["Variables"][0]["Initial Standard Deviation"] = 0.25;
   e["Variables"][0]["Lower Bound"] = -1.0;
   e["Variables"][0]["Upper Bound"] = 1.0;

   // Creating optimizer configuration Json
   knlohmann::json samplerJs;
   samplerJs["Type"] = "Sampler/MCMC";

   // Creating module
   MCMC* sampler;
   ASSERT_NO_THROW(sampler = dynamic_cast<MCMC *>(Module::getModule(samplerJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(sampler->applyModuleDefaults(samplerJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(sampler->applyVariableDefaults());

   // Base solver class methods
   ASSERT_NO_THROW(sampler->Solver::printGenerationBefore());
   ASSERT_NO_THROW(sampler->Solver::printGenerationAfter());
   ASSERT_NO_THROW(sampler->Solver::setInitialConfiguration());

   // Backup the correct base configuration
   auto baseOptJs = samplerJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   // Testing optional parameters
   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Variable Count"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Variable Count"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Model Evaluation Count"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Model Evaluation Count"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Lower Bound");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Lower Bound"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Lower Bound"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Upper Bound");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Upper Bound"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Upper Bound"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Initial Value");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Value"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Value"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   Sample s;
   Module* m = sampler;
   knlohmann::json js;
   ASSERT_NO_THROW(m->Module::initialize());
   ASSERT_NO_THROW(m->Module::finalize());
   ASSERT_NO_THROW(m->Module::getType());
   ASSERT_NO_THROW(m->Module::checkTermination());
   ASSERT_NO_THROW(m->Module::applyVariableDefaults());
   ASSERT_NO_THROW(m->Module::getConfiguration(js));
   ASSERT_NO_THROW(m->Module::setConfiguration(js));
   ASSERT_NO_THROW(m->Module::applyModuleDefaults(js));
  }

  //////////////// MCMC CLASS ////////////////////////

  TEST(samplers, MCMC)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();

   // Creating initial variable
   Variable v;
   e._variables.push_back(&v);
   e["Variables"][0]["Name"] = "Var 1";
   e["Variables"][0]["Initial Mean"] = 0.0;
   e["Variables"][0]["Initial Standard Deviation"] = 0.25;
   e["Variables"][0]["Lower Bound"] = -1.0;
   e["Variables"][0]["Upper Bound"] = 1.0;

   // Creating optimizer configuration Json
   knlohmann::json samplerJs;
   samplerJs["Type"] = "Sampler/MCMC";

   // Creating module
   MCMC* sampler;
   ASSERT_NO_THROW(sampler = dynamic_cast<MCMC *>(Module::getModule(samplerJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(sampler->applyModuleDefaults(samplerJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(sampler->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = samplerJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   auto inMat = std::vector<double>({1.0, 2.0, 2.0, 13.0});
   auto outMat = std::vector<double>({1.0, 2.0, 2.0, 13.0});
   sampler->_variableCount = 2;
   ASSERT_NO_THROW(sampler->choleskyDecomp(inMat, outMat));

   // Testing optional parameters
   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Cholesky Decomposition Covariance"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Cholesky Decomposition Covariance"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Cholesky Decomposition Chain Covariance"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Cholesky Decomposition Chain Covariance"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leader Evaluation"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leader Evaluation"] = 0.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Evaluations"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Evaluations"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Rejection Alphas"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Rejection Alphas"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Rate"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Rate"] = 0.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Count"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Count"] = 0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Proposed Sample Count"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Proposed Sample Count"] = 0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Database"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Database"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leader"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leader"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidate"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidate"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Evaluation Database"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Evaluation Database"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Mean"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Mean"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Covariance Placeholder"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Covariance Placeholder"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Covariance"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Covariance"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Length"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Length"] = 0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Burn In");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Burn In"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Burn In"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Leap");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Leap"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Leap"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Rejection Levels");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Rejection Levels"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Rejection Levels"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Use Adaptive Sampling");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Use Adaptive Sampling"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Use Adaptive Sampling"] = true;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Non Adaption Period");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Non Adaption Period"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Non Adaption Period"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Chain Covariance Scaling");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Covariance Scaling"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Covariance Scaling"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"].erase("Max Samples");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Samples"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Samples"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"].erase("Max Model Evaluations");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Model Evaluations"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Model Evaluations"] = 10000;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"].erase("Max Generations");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Generations"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Generations"] = 10000;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   ///// Variable Tests

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Initial Mean");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Mean"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Mean"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Initial Standard Deviation");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Standard Deviation"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Standard Deviation"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

  }

  //////////////// TMCMC CLASS ////////////////////////

  TEST(samplers, TMCMC)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();

   // Creating initial variable
   Variable v;
   v._distributionIndex = 0;
   e._variables.push_back(&v);
   e["Variables"][0]["Name"] = "Var 1";
   e["Variables"][0]["Initial Mean"] = 0.0;
   e["Variables"][0]["Initial Standard Deviation"] = 0.25;
   e["Variables"][0]["Lower Bound"] = -1.0;
   e["Variables"][0]["Prior Distribution"] = "Uniform 0";

   e["Distributions"][0]["Name"] = "Uniform 0";
   e["Distributions"][0]["Type"] = "Univariate/Uniform";
   e["Distributions"][0]["Minimum"] = 0.0;
   e["Distributions"][0]["Maximum"] = +5.0;

   knlohmann::json uniformDistroJs;
   uniformDistroJs["Type"] = "Univariate/Uniform";
   uniformDistroJs["Minimum"] = 0.0;
   uniformDistroJs["Maximum"] = 1.0;
   auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
   uniformGenerator->applyVariableDefaults();
   uniformGenerator->applyModuleDefaults(uniformDistroJs);
   uniformGenerator->setConfiguration(uniformDistroJs);
   e._distributions.push_back(uniformGenerator);

   // Configuring Problem
   e["Problem"]["Type"] = "Bayesian/Reference";
   Reference* pR;
   knlohmann::json problemRefJs;
   problemRefJs["Type"] = "Bayesian/Reference";
   ASSERT_NO_THROW(pR = dynamic_cast<Reference *>(Module::getModule(problemRefJs, &e)));
   pR->_likelihoodModel = "Normal";
   pR->_referenceData = std::vector<double>({0.5});
   e._problem = pR;

   // Creating optimizer configuration Json
   knlohmann::json samplerJs;
   samplerJs["Type"] = "Sampler/TMCMC";
   samplerJs["Population Size"] = 512;

   // Creating module
   TMCMC* sampler;
   ASSERT_NO_THROW(sampler = dynamic_cast<TMCMC *>(Module::getModule(samplerJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(sampler->applyModuleDefaults(samplerJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(sampler->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = samplerJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   // Test version compatbility
   sampler->_version = "mTMCMC";
   e["Problem"]["Type"] = "Bayesian/Custom";
   ASSERT_ANY_THROW(sampler->setInitialConfiguration());
   e["Problem"]["Type"] = "Bayesian/Reference";
   ASSERT_NO_THROW(sampler->setInitialConfiguration());

   // Testing optional parameters
   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Current Burn In"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Current Burn In"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Pending Evaluation"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Pending Evaluation"] = std::vector<int>({0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Pending Gradient"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Pending Gradient"] = std::vector<int>({0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates LogLikelihoods"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates LogLikelihoods"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates LogPriors"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates LogPriors"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Gradients"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Gradients"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Errors"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Errors"] = std::vector<int>({0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Covariance"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Candidates Covariance"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders LogLikelihoods"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders LogLikelihoods"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders LogPriors"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders LogPriors"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders Covariance"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders Covariance"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders Gradients"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders Gradients"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders Errors"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Leaders Errors"] = std::vector<int>({0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Finished Chains Count"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Finished Chains Count"] = 0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Current Chain Step"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Current Chain Step"] = std::vector<int>({0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Lengths"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Lengths"] = std::vector<int>({0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Coefficient Of Variation"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Coefficient Of Variation"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Count"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Count"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Annealing Exponent"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Annealing Exponent"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Previous Annealing Exponent"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Previous Annealing Exponent"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Finite Prior Evaluations"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Finite Prior Evaluations"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Finite Likelihood Evaluations"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Finite Likelihood Evaluations"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Accepted Samples Count"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Accepted Samples Count"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LogEvidence"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LogEvidence"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Proposals Acceptance Rate"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Proposals Acceptance Rate"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Selection Acceptance Rate"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Selection Acceptance Rate"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Covariance Matrix"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Covariance Matrix"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Loglikelihood"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Loglikelihood"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Mean Theta"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Mean Theta"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Database"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Database"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample LogLikelihood Database"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample LogLikelihood Database"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample LogPrior Database"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample LogPrior Database"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Gradient Database"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Gradient Database"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Error Database"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Error Database"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Covariances Database"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Covariances Database"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Upper Extended Boundaries"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Upper Extended Boundaries"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Lower Extended Boundaries"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Lower Extended Boundaries"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num LU Decomposition Failures Proposal"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num LU Decomposition Failures Proposal"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Eigen Decomposition Failures Proposal"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Eigen Decomposition Failures Proposal"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Inversion Failures Proposal"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Inversion Failures Proposal"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Negative Definite Proposals"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Negative Definite Proposals"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Cholesky Decomposition Failures Proposal"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Cholesky Decomposition Failures Proposal"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Covariance Corrections"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Covariance Corrections"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   // Testing mandatory values

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Version"] = 1;
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Version"] = "Undefined";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Version");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Version"] = "TMCMC";
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Version"] = "mTMCMC";
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Population Size");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Population Size"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Population Size"] = 512;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Max Chain Length");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Chain Length"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Chain Length"] = 16;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Burn In");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Burn In"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Burn In"] = 4;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Per Generation Burn In");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Per Generation Burn In"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Per Generation Burn In"] = std::vector<size_t>({4});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Target Coefficient Of Variation");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Target Coefficient Of Variation"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Target Coefficient Of Variation"] = 4;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Covariance Scaling");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Covariance Scaling"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Covariance Scaling"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Min Annealing Exponent Update");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Min Annealing Exponent Update"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Min Annealing Exponent Update"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Max Annealing Exponent Update");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Annealing Exponent Update"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Annealing Exponent Update"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Step Size");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Step Size"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Step Size"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Domain Extension Factor");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Domain Extension Factor"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Domain Extension Factor"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"].erase("Target Annealing Exponent");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Target Annealing Exponent"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Target Annealing Exponent"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));
  }

  //////////////// HMC CLASS ////////////////////////

  TEST(samplers, HMC)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();
   e._logger = new Logger("Detailed", stdout);

   // Creating initial variable
   Variable v;
   e._variables.push_back(&v);
   v._distributionIndex = 0;
   e["Variables"][0]["Name"] = "Var 1";
   e["Variables"][0]["Initial Mean"] = 0.0;
   e["Variables"][0]["Initial Standard Deviation"] = 0.25;
   e["Variables"][0]["Lower Bound"] = -1.0;
   e["Variables"][0]["Upper Bound"] = 1.0;
   e["Variables"][0]["Prior Distribution"] = "Uniform 0";

   e["Distributions"][0]["Name"] = "Uniform 0";
   e["Distributions"][0]["Type"] = "Univariate/Uniform";
   e["Distributions"][0]["Minimum"] = 0.0;
   e["Distributions"][0]["Maximum"] = +5.0;

   knlohmann::json uniformDistroJs;
   uniformDistroJs["Type"] = "Univariate/Uniform";
   uniformDistroJs["Minimum"] = 0.0;
   uniformDistroJs["Maximum"] = 1.0;
   auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
   uniformGenerator->applyVariableDefaults();
   uniformGenerator->applyModuleDefaults(uniformDistroJs);
   uniformGenerator->setConfiguration(uniformDistroJs);
   e._distributions.push_back(uniformGenerator);

   // Configuring Problem
   e["Problem"]["Type"] = "Sampling";

   // Creating sampling problem module
   Sampling* pS;
   knlohmann::json problemJs;
   problemJs["Type"] = "Sampling";
   ASSERT_NO_THROW(pS = dynamic_cast<Sampling *>(Module::getModule(problemJs, &e)));
   e._problem = pS;

   Reference* pR;
   knlohmann::json problemRefJs;
   problemRefJs["Type"] = "Bayesian/Reference";
   ASSERT_NO_THROW(pR = dynamic_cast<Reference *>(Module::getModule(problemRefJs, &e)));
   pR->_likelihoodModel = "Normal";
   pR->_referenceData = std::vector<double>({0.5});

   // Creating optimizer configuration Json
   knlohmann::json samplerJs;
   samplerJs["Type"] = "Sampler/HMC";

   // Creating module
   HMC* sampler;
   ASSERT_NO_THROW(sampler = dynamic_cast<HMC *>(Module::getModule(samplerJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(sampler->applyModuleDefaults(samplerJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(sampler->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = samplerJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));
   sampler->_variableCount = 2;

   // Test initial configuration

   sampler->_version = "Riemannian";
   sampler->_useDiagonalMetric = false;
   ASSERT_ANY_THROW(sampler->setInitialConfiguration());

   sampler->_version = "Undefined";
   ASSERT_ANY_THROW(sampler->setInitialConfiguration());
   sampler->_version = "Static";

   sampler->_version = "Euclidean";
   sampler->_useAdaptiveStepSize = true;
   sampler->_burnIn = 200;
   sampler->_initialFastAdaptionInterval = 300;
   ASSERT_NO_THROW(sampler->setInitialConfiguration());

   sampler->_version = "Euclidean";
   sampler->_useAdaptiveStepSize = true;
   sampler->_burnIn = 0;
   sampler->_initialFastAdaptionInterval = 0;
   ASSERT_ANY_THROW(sampler->setInitialConfiguration());

   v._initialMean = std::numeric_limits<double>::infinity();
   ASSERT_ANY_THROW(sampler->setInitialConfiguration());
   v._initialMean = 0.0;

   v._initialStandardDeviation = std::numeric_limits<double>::infinity();
   ASSERT_ANY_THROW(sampler->setInitialConfiguration());
   v._initialStandardDeviation = 0.0;

   // Testing optional parameters
   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Metric Type"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Metric Type"] = Metric::Static;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Rate"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Rate"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Running Acceptance Rate"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Running Acceptance Rate"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Proposed Sample Count"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Proposed Sample Count"] = 0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Database"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Database"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Euclidean Warmup Sample Database"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Euclidean Warmup Sample Database"] = std::vector<std::vector<double>>({{0.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Evaluation Database"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sample Evaluation Database"] = std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Length"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Chain Length"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Leader Evaluation"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Leader Evaluation"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Candidate Evaluation"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Candidate Evaluation"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Position Leader"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Position Leader"] =  std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Position Candidate"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Position Candidate"] =  std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Momentum Leader"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Momentum Leader"] =  std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Momentum Candidate"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Momentum Candidate"] =  std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Log Dual Step Size"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Log Dual Step Size"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Mu"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Mu"] =  1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["H Bar"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["H Bar"] =  1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Count NUTS"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Count NUTS"] =  1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Count"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Count"] =  1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Current Depth"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Current Depth"] =  1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Probability"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Probability"] =  1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Rate Error"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Rate Error"] =  1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Metric"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Metric"] =  std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Inverse Metric"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Inverse Metric"] =  std::vector<double>({0.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   // Testing mandatory values

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Burn In");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Burn In"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Burn In"] = 4;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Use Diagonal Metric");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Use Diagonal Metric"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Use Diagonal Metric"] = true;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Num Integration Steps");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Integration Steps"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Num Integration Steps"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Max Integration Steps");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Integration Steps"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Integration Steps"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Use NUTS");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Use NUTS"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Use NUTS"] = true;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Step Size");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Step Size"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Step Size"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Use Adaptive Step Size");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Use Adaptive Step Size"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Use Adaptive Step Size"] = true;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Target Acceptance Rate");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Target Acceptance Rate"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Target Acceptance Rate"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Acceptance Rate Learning Rate");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Rate Learning Rate"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Acceptance Rate Learning Rate"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Target Integration Time");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Target Integration Time"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Target Integration Time"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Adaptive Step Size Speed Constant");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Adaptive Step Size Speed Constant"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Adaptive Step Size Speed Constant"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Adaptive Step Size Stabilization Constant");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Adaptive Step Size Stabilization Constant"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Adaptive Step Size Stabilization Constant"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Adaptive Step Size Schedule Constant");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Adaptive Step Size Schedule Constant"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Adaptive Step Size Schedule Constant"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Max Depth");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Depth"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Depth"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Version"] = 1;
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Version");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Version"] = "Static";
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Inverse Regularization Parameter");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Inverse Regularization Parameter"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Inverse Regularization Parameter"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Max Fixed Point Iterations");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Fixed Point Iterations"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Fixed Point Iterations"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Step Size Jitter");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Step Size Jitter"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Step Size Jitter"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Initial Fast Adaption Interval");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Initial Fast Adaption Interval"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Initial Fast Adaption Interval"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Final Fast Adaption Interval");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Final Fast Adaption Interval"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Final Fast Adaption Interval"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Initial Slow Adaption Interval");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Initial Slow Adaption Interval"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Initial Slow Adaption Interval"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"].erase("Max Samples");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Samples"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Samples"] = 32;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   ///// Variable Tests

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Initial Mean");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Mean"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Mean"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Initial Standard Deviation");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Standard Deviation"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Standard Deviation"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   ///// Testing Helpers
   auto unitVec = std::vector<double>({1.0});
   auto unitMat = std::vector<std::vector<double>>({{1.0}});
   Hamiltonian* h;
   e._overrideEngine = true;
   e._overrideFunction = [](Sample &s)
   {
    s["F(x)"] = 0.5;
    s["logP(x)"] = 0.5;
    s["logLikelihood"] = 0.5;
    s["logLikelihood Gradient"] = std::vector<double>({0.5});
    s["grad(logP(x))"] = std::vector<double>({0.5});
    s["H(logP(x))"] = std::vector<std::vector<double>>({{0.5}});
    s["Reference Evaluations"] = std::vector<double>({0.5});
    s["Standard Deviation"] = std::vector<double>({0.5});
    s["Gradient Mean"] = std::vector<std::vector<double>>({{0.5}});
    s["Gradient Standard Deviation"] = std::vector<std::vector<double>>({{0.5}});
    s["Hessian Mean"] = std::vector<std::vector<double>>({{0.5}});
    s["Hessian Standard Deviation"] = std::vector<std::vector<double>>({{0.5}});
   };

   ASSERT_NO_THROW(h = new HamiltonianEuclideanDense(1, sampler->_multivariateGenerator, unitVec, &e));
   ASSERT_NO_THROW(h->H(unitVec, unitVec));
   ASSERT_NO_THROW(h->dtau_dq(unitVec, unitVec));
   ASSERT_NO_THROW(h->dtau_dp(unitVec, unitVec));
   ASSERT_NO_THROW(h->tau(unitVec, unitVec));
   ASSERT_NO_THROW(h->phi());
   ASSERT_NO_THROW(h->dphi_dq());
   ASSERT_NO_THROW(h->innerProduct(unitVec, unitVec, unitVec));
   ASSERT_NO_THROW(h->computeStandardCriterion(unitVec, unitVec, unitVec, unitVec));
   ASSERT_NO_THROW(h->sampleMomentum(unitVec));
   ASSERT_NO_THROW(delete h);

   auto squareVec0 = std::vector<double>({1.0, -0.3, 0.5, 0.8});
   auto squareVec1 = std::vector<double>({1.0, -0.3, 0.5, 0.8});
   auto squaretMat = std::vector<std::vector<double>>({{1.0, -0.3}, {0.5, 0.8}});
   ASSERT_NO_THROW(h = new HamiltonianEuclideanDense(2, sampler->_multivariateGenerator, squareVec0, &e));
   h->updateMetricMatricesEuclidean(squaretMat, squareVec0, squareVec1);
   ASSERT_NO_THROW(h->updateMetricMatricesEuclidean(squaretMat, squareVec0, squareVec1));
   ASSERT_NO_THROW(delete h);

   ASSERT_NO_THROW(h = new HamiltonianEuclideanDense(1, sampler->_multivariateGenerator, unitVec, &e));
   h->_k = &e;
   h->samplingProblemPtr = pS;
   h->updateHamiltonian(unitVec,unitVec,unitVec);
   ASSERT_NO_THROW(delete h);

   ASSERT_NO_THROW(h = new HamiltonianEuclideanDiag(1, sampler->_normalGenerator, &e));
   ASSERT_NO_THROW(h->H(unitVec, unitVec));
   ASSERT_NO_THROW(h->K(unitVec, unitVec));
   ASSERT_NO_THROW(h->dK(unitVec, unitVec));
   ASSERT_NO_THROW(h->sampleMomentum(unitVec));
   ASSERT_NO_THROW(h->innerProduct(unitVec, unitVec, unitVec));
   ASSERT_NO_THROW(h->computeStandardCriterion(unitVec, unitVec, unitVec, unitVec));
   ASSERT_NO_THROW(h->updateMetricMatricesRiemannian(unitVec, unitVec));
   ASSERT_NO_THROW(h->updateMetricMatricesEuclidean(unitMat, unitVec, unitVec));
   h->_k = &e;
   h->samplingProblemPtr = pS;
   h->bayesianProblemPtr = NULL;
   ASSERT_NO_THROW(h->updateHamiltonian(unitVec,unitVec,unitVec));
   h->samplingProblemPtr = NULL;
   h->bayesianProblemPtr = pR;
   ASSERT_NO_THROW(h->updateHamiltonian(unitVec,unitVec,unitVec));
   ASSERT_NO_THROW(delete h);

   ASSERT_NO_THROW(h = new HamiltonianRiemannianConstDense(1, sampler->_multivariateGenerator, unitVec, 1.0, &e));
   ASSERT_NO_THROW(h->H(unitVec, unitVec));
   ASSERT_NO_THROW(h->dtau_dq(unitVec, unitVec));
   ASSERT_NO_THROW(h->dtau_dp(unitVec, unitVec));
   ASSERT_NO_THROW(h->phi());
   ASSERT_NO_THROW(h->dphi_dq());
   ASSERT_NO_THROW(h->sampleMomentum(unitVec));
   ASSERT_NO_THROW(h->innerProduct(unitVec, unitVec, unitVec));
   ASSERT_NO_THROW(h->computeStandardCriterion(unitVec, unitVec, unitVec, unitVec));
   ASSERT_NO_THROW(h->updateMetricMatricesRiemannian(unitVec, unitVec));
   ASSERT_NO_THROW(h->updateMetricMatricesEuclidean(unitMat, unitVec, unitVec));
   h->_k = &e;
   h->samplingProblemPtr = pS;
   h->bayesianProblemPtr = NULL;
   ASSERT_NO_THROW(h->updateHamiltonian(unitVec,unitVec,unitVec));
   h->samplingProblemPtr = NULL;
   h->bayesianProblemPtr = pR;
   ASSERT_NO_THROW(h->updateHamiltonian(unitVec,unitVec,unitVec));
   ASSERT_NO_THROW(delete h);

   ASSERT_NO_THROW(h = new HamiltonianRiemannianConstDiag(1, sampler->_normalGenerator, 1.0, &e));
   ASSERT_NO_THROW(h->H(unitVec, unitVec));
   ASSERT_NO_THROW(h->K(unitVec, unitVec));
   ASSERT_NO_THROW(h->dK(unitVec, unitVec));
   ASSERT_NO_THROW(h->tau(unitVec, unitVec));
   ASSERT_NO_THROW(h->dtau_dq(unitVec, unitVec));
   ASSERT_NO_THROW(h->dtau_dp(unitVec, unitVec));
   ASSERT_NO_THROW(h->phi());
   ASSERT_NO_THROW(h->dphi_dq());
   ASSERT_NO_THROW(h->sampleMomentum(unitVec));
   ASSERT_NO_THROW(h->innerProduct(unitVec, unitVec, unitVec));
   ASSERT_NO_THROW(h->computeStandardCriterion(unitVec, unitVec, unitVec, unitVec));
   ASSERT_NO_THROW(h->updateMetricMatricesRiemannian(unitVec, unitVec));
   ASSERT_NO_THROW(h->updateMetricMatricesEuclidean(unitMat, unitVec, unitVec));
   h->_k = &e;
   h->samplingProblemPtr = pS;
   h->bayesianProblemPtr = NULL;
   ASSERT_NO_THROW(h->updateHamiltonian(unitVec,unitVec,unitVec));
   h->samplingProblemPtr = NULL;
   h->bayesianProblemPtr = pR;
   ASSERT_NO_THROW(h->updateHamiltonian(unitVec,unitVec,unitVec));
   ASSERT_NO_THROW(delete h);

   ASSERT_NO_THROW(h = new HamiltonianRiemannianDiag(1, sampler->_normalGenerator, 1.0, &e));
   h->_currentGradient = unitVec;
   ASSERT_NO_THROW(h->H(unitVec, unitVec));
   ASSERT_NO_THROW(h->K(unitVec, unitVec));
   ASSERT_NO_THROW(h->dK(unitVec, unitVec));
   ASSERT_NO_THROW(h->tau(unitVec, unitVec));
   ASSERT_NO_THROW(h->dtau_dq(unitVec, unitVec));
   ASSERT_NO_THROW(h->dtau_dp(unitVec, unitVec));
   ASSERT_NO_THROW(h->phi());
   ASSERT_NO_THROW(h->dphi_dq());
   ASSERT_NO_THROW(h->sampleMomentum(unitVec));
   ASSERT_NO_THROW(h->innerProduct(unitVec, unitVec, unitVec));
   ASSERT_NO_THROW(h->computeStandardCriterion(unitVec, unitVec, unitVec, unitVec));
   ASSERT_NO_THROW(h->updateMetricMatricesRiemannian(unitVec, unitVec));
   ASSERT_NO_THROW(h->updateMetricMatricesEuclidean(unitMat, unitVec, unitVec));
   ASSERT_NO_THROW(((HamiltonianRiemannianDiag*)h)->softAbsFunc(0.0, 0.0));
   ASSERT_NO_THROW(((HamiltonianRiemannianDiag*)h)->softAbsFunc(1.0, 1.0));
   h->_k = &e;
   h->samplingProblemPtr = pS;
   h->bayesianProblemPtr = NULL;
   ASSERT_NO_THROW(h->updateHamiltonian(unitVec,unitVec,unitVec));
   h->samplingProblemPtr = NULL;
   h->bayesianProblemPtr = pR;
   ASSERT_NO_THROW(h->updateHamiltonian(unitVec,unitVec,unitVec));

   ASSERT_NO_THROW(((HamiltonianRiemannianDiag*)h)->taylorSeriesPhiFunc(0.1, 0.1));
   ASSERT_NO_THROW(((HamiltonianRiemannianDiag*)h)->taylorSeriesPhiFunc(1.0, 1.0));
   ASSERT_NO_THROW(((HamiltonianRiemannianDiag*)h)->taylorSeriesTauFunc(0.1, 0.1));
   ASSERT_NO_THROW(((HamiltonianRiemannianDiag*)h)->taylorSeriesTauFunc(1.0, 1.0));

   TreeHelperEuclidean tE;
   ASSERT_NO_THROW(tE.computeCriterion(*h));
   ASSERT_ANY_THROW(tE.computeCriterion(*h, unitVec, unitVec, unitVec, unitVec));

   TreeHelperRiemannian tR;
   ASSERT_ANY_THROW(tR.computeCriterion(*h));
   ASSERT_NO_THROW(tR.computeCriterion(*h, unitVec, unitVec, unitVec, unitVec));

   std::shared_ptr<HamiltonianRiemannianDiag> sh;
   ASSERT_NO_THROW(sh = std::make_shared<HamiltonianRiemannianDiag>(HamiltonianRiemannianDiag(1, sampler->_normalGenerator, 1.0, &e)));
   LeapfrogExplicit  *leap;
   ASSERT_NO_THROW(leap = new LeapfrogExplicit(sh));
   ASSERT_NO_THROW(leap->step(unitVec, unitVec, unitVec, unitVec, 1.0));
   ASSERT_NO_THROW(delete leap);

   ASSERT_NO_THROW(delete h);

   e._problem = nullptr;
   ASSERT_ANY_THROW(sh = std::make_shared<HamiltonianRiemannianDiag>(HamiltonianRiemannianDiag(1, sampler->_normalGenerator, 1.0, &e)));
   e._problem = pS;
  }

  //////////////// Nested CLASS ////////////////////////

  TEST(samplers, Nested)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();

   // Creating initial variable
   Variable v;
   v._distributionIndex = 0;
   e._variables.push_back(&v);
   e["Variables"][0]["Name"] = "Var 1";
   e["Variables"][0]["Initial Mean"] = 0.0;
   e["Variables"][0]["Initial Standard Deviation"] = 0.25;
   e["Variables"][0]["Lower Bound"] = -1.0;
   e["Variables"][0]["Upper Bound"] = 1.0;
   e["Variables"][0]["Prior Distribution"] = "Uniform 0";

   e["Distributions"][0]["Name"] = "Uniform 0";
   e["Distributions"][0]["Type"] = "Univariate/Uniform";
   e["Distributions"][0]["Minimum"] = 0.0;
   e["Distributions"][0]["Maximum"] = +5.0;

   knlohmann::json normalDistroJs;
   normalDistroJs["Type"] = "Univariate/Normal";
   normalDistroJs["Mean"] = 0.0;
   normalDistroJs["Standard Deviation"] = 1.0;
   auto normalGenerator = dynamic_cast<korali::distribution::univariate::Normal*>(korali::Module::getModule(normalDistroJs, &e));
   normalGenerator->applyVariableDefaults();
   normalGenerator->applyModuleDefaults(normalDistroJs);
   normalGenerator->setConfiguration(normalDistroJs);
   e._distributions.push_back(normalGenerator);

   // Configuring Problem
   e["Problem"]["Type"] = "Bayesian/Reference";
   Reference* pR;
   knlohmann::json problemRefJs;
   problemRefJs["Type"] = "Bayesian/Reference";
   ASSERT_NO_THROW(pR = dynamic_cast<Reference *>(Module::getModule(problemRefJs, &e)));
   pR->_likelihoodModel = "Normal";
   pR->_referenceData = std::vector<double>({0.5});
   e._problem = pR;

   // Creating optimizer configuration Json
   knlohmann::json samplerJs;
   samplerJs["Type"] = "Sampler/Nested";
   samplerJs["Resampling Method"] = "Box";

   // Creating module
   Nested* sampler;
   ASSERT_NO_THROW(sampler = dynamic_cast<Nested *>(Module::getModule(samplerJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(sampler->applyModuleDefaults(samplerJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(sampler->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = samplerJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   // Trying initial configuration up optimizer correctly
   ASSERT_NO_THROW(sampler->setInitialConfiguration());

   // Testing optional parameters
   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Accepted Samples"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Accepted Samples"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Generated Samples"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Generated Samples"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LogEvidence"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LogEvidence"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LogEvidence Var"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LogEvidence Var"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LogVolume"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LogVolume"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Bound LogVolume"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Bound LogVolume"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Last Accepted"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Last Accepted"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Next Update"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Next Update"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Information"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Information"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LStar"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LStar"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LStarOld"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LStarOld"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LogWeight"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["LogWeight"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Expected LogShrinkage"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Expected LogShrinkage"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Evaluation"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Max Evaluation"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Remaining Log Evidence"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Remaining Log Evidence"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Log Evidence Difference"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Log Evidence Difference"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Effective Sample Size"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Effective Sample Size"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sum Log Weights"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sum Log Weights"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sum Square Log Weights"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Sum Square Log Weights"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Prior Lower Bound"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Prior Lower Bound"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Prior Width"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Prior Width"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Candidates"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Candidates"] = std::vector<std::vector<double>>({{1.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Candidate LogLikelihoods"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Candidate LogLikelihoods"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Candidate LogPriors"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Candidate LogPriors"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Candidate LogPrior Weights"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Candidate LogPrior Weights"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Live Samples"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Live Samples"] = std::vector<std::vector<double>>({{1.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Live LogLikelihoods"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Live LogLikelihoods"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Live LogPriors"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Live LogPriors"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Live LogPrior Weights"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Live LogPrior Weights"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Live Samples Rank"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Live Samples Rank"] = std::vector<double>({1});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Number Dead Samples"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Number Dead Samples"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Dead Samples"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Dead Samples"] = std::vector<std::vector<double>>({{1.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Dead LogLikelihoods"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Dead LogLikelihoods"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Dead LogPriors"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Dead LogPriors"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Dead LogPrior Weights"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Dead LogPrior Weights"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Dead LogWeights"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Dead LogWeights"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Covariance Matrix"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Covariance Matrix"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Log Domain Size"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Log Domain Size"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Domain Mean"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Domain Mean"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Box Lower Bound"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Box Lower Bound"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Box Upper Bound"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Box Upper Bound"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Ellipse Axes"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Ellipse Axes"] = std::vector<std::vector<double>>({{1.0}});
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Number Live Points"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Number Live Points");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Number Live Points"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Number Live Points"] = 512;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Batch Size");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Batch Size"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Batch Size"] = 512;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Add Live Points");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Add Live Points"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Add Live Points"] = 512;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Resampling Method");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Resampling Method"] = 32;
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Resampling Method"] = "Box";
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Proposal Update Frequency");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Proposal Update Frequency"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Proposal Update Frequency"] = 512;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs.erase("Ellipsoidal Scaling");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Ellipsoidal Scaling"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Ellipsoidal Scaling"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"].erase("Min Log Evidence Delta");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Min Log Evidence Delta"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Min Log Evidence Delta"] = 1.0;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"].erase("Max Effective Sample Size");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Effective Sample Size"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Effective Sample Size"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"].erase("Max Log Likelihood");
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Log Likelihood"] = "Not a Number";
   ASSERT_ANY_THROW(sampler->setConfiguration(samplerJs));

   samplerJs = baseOptJs;
   experimentJs = baseExpJs;
   samplerJs["Termination Criteria"]["Max Log Likelihood"] = 1;
   ASSERT_NO_THROW(sampler->setConfiguration(samplerJs));

   // Testing individual functions
   ellipse_t ellipse(1);
   ASSERT_NO_THROW(ellipse.initSphere());
   ASSERT_NO_THROW(ellipse.scaleVolume(1.0));

   // Testing termination criteria
   e._currentGeneration = 2;
   sampler->_lStar = 1.0;

   sampler->_maxLogLikelihood = 2.0;
   ASSERT_FALSE(sampler->checkTermination());
   sampler->_maxLogLikelihood = 0.5;
   ASSERT_TRUE(sampler->checkTermination());
   sampler->_maxLogLikelihood= 2.0;
  }

} // namespace
