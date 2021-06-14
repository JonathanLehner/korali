#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/problem/optimization/optimization.hpp"
#include "modules/problem/sampling/sampling.hpp"
#include "sample/sample.hpp"

namespace
{
 using namespace korali;
 using namespace korali::problem;

 TEST(Problem, Optimization)
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
  // Configuring Problem
  e["Problem"]["Type"] = "Optimization";
  Optimization* pO;
  knlohmann::json problemJs;
  problemJs["Type"] = "Optimization";
  problemJs["Objective Function"] = 0;
  e["Solver"]["Type"] = "Optimizer/CMAES";

  ASSERT_NO_THROW(pO = dynamic_cast<Optimization *>(Module::getModule(problemJs, &e)));
  e._problem = pO;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pO->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pO->applyVariableDefaults();
  ASSERT_NO_THROW(pO->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseOptJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pO->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

  // Evaluation function
  std::function<void(korali::Sample&)> modelFc = [](Sample& s)
  {
   s["F(x)"] = 1.0;
   s["Gradient"] = std::vector<double>({ 1.0 });
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);
  pO->_constraints = std::vector<size_t>({0});

  // Evaluating correct execution of evaluation
  Sample s;
  ASSERT_NO_THROW(pO->evaluateConstraints(s));
  ASSERT_NO_THROW(pO->evaluate(s));
  ASSERT_NO_THROW(pO->evaluateWithGradients(s));

  // Evaluating incorrect execution of evaluation
  modelFc = [](Sample& s)
  {
   s["F(x)"] = std::numeric_limits<double>::infinity();
   s["Gradient"] = std::vector<double>(1.0);
  };

  ASSERT_ANY_THROW(pO->evaluateConstraints(s));
  ASSERT_ANY_THROW(pO->evaluate(s));
  ASSERT_ANY_THROW(pO->evaluateWithGradients(s));

  // Evaluating incorrect execution of gradient
  modelFc = [](Sample& s)
  {
   s["F(x)"] = 1.0;
   s["Gradient"] = std::vector<double>({ std::numeric_limits<double>::infinity() });
  };

  ASSERT_NO_THROW(pO->evaluateConstraints(s));
  ASSERT_NO_THROW(pO->evaluate(s));
  ASSERT_ANY_THROW(pO->evaluateWithGradients(s));

  // Evaluating correct execution of multiple evaluations
  modelFc = [](Sample& s)
  {
   s["F(x)"] = std::vector<double>({1.0, 1.0});
  };

  ASSERT_NO_THROW(pO->evaluateMultiple(s));

  // Trying to run unknown operation
  ASSERT_ANY_THROW(pO->runOperation("Unknown", s));

  // Evaluating incorrect execution of multiple evaluations
  modelFc = [](Sample& s)
  {
   s["F(x)"] = std::vector<double>({std::numeric_limits<double>::infinity(), 1.0});
  };

  ASSERT_ANY_THROW(pO->evaluateMultiple(s));

  // Testing optional parameters
  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Has Discrete Variables"] = "Not a Number";
  ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Has Discrete Variables"] = true;
  ASSERT_NO_THROW(pO->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs.erase("Num Objectives");
  ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Num Objectives"] = "Not a Number";
  ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Num Objectives"] = 1;
  ASSERT_NO_THROW(pO->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs.erase("Objective Function");
  ASSERT_ANY_THROW(pO->setConfiguration(problemJs));


  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Objective Function"] = "Not a Number";
  ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Objective Function"] = 1;
  ASSERT_NO_THROW(pO->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs.erase("Constraints");
  ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Constraints"] = "Not a Number";
  ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Constraints"] = std::vector<uint64_t>({1});
  ASSERT_NO_THROW(pO->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Granularity");
  ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Granularity"] = "Not a Number";
  ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Granularity"] = 1.0;
  ASSERT_NO_THROW(pO->setConfiguration(problemJs));
 };

 TEST(Problem, Sampling)
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

  // Configuring Problem
  e["Problem"]["Type"] = "Sampling";
  Sampling* pS;
  knlohmann::json problemJs;
  problemJs["Type"] = "Sampling";
  problemJs["Probability Function"] = 0;
  e["Solver"]["Type"] = "Sampler/MCMC";

  ASSERT_NO_THROW(pS = dynamic_cast<Sampling *>(Module::getModule(problemJs, &e)));
  e._problem = pS;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pS->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pS->applyVariableDefaults();
  ASSERT_NO_THROW(pS->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseSamJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pS->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseSamJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pS->setConfiguration(problemJs));

  // Evaluation function
  std::function<void(korali::Sample&)> modelFc = [](Sample& s)
  {
   s["logP(x)"] = 0.5;
   s["grad(logP(x))"] = std::vector<double>({0.5});
   s["H(logP(x))"] = std::vector<std::vector<double>>({{0.5}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  // Evaluating correct execution of evaluation
  Sample s;
  ASSERT_NO_THROW(pS->evaluate(s));
  ASSERT_NO_THROW(pS->evaluateGradient(s));
  ASSERT_NO_THROW(pS->evaluateHessian(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logP(x)"] = std::numeric_limits<double>::infinity();
   s["grad(logP(x))"] = std::vector<double>({0.5});
   s["H(logP(x))"] = std::vector<std::vector<double>>({{0.5}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  ASSERT_ANY_THROW(pS->evaluate(s));
  ASSERT_NO_THROW(pS->evaluateGradient(s));
  ASSERT_NO_THROW(pS->evaluateHessian(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logP(x)"] = 0.5;
   s["grad(logP(x))"] = std::vector<double>({});
   s["H(logP(x))"] = std::vector<std::vector<double>>({{0.5}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  ASSERT_NO_THROW(pS->evaluate(s));
  ASSERT_ANY_THROW(pS->evaluateGradient(s));
  ASSERT_NO_THROW(pS->evaluateHessian(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logP(x)"] = 0.5;
   s["grad(logP(x))"] = std::vector<double>({0.5});
   s["H(logP(x))"] = std::vector<std::vector<double>>({{}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  ASSERT_NO_THROW(pS->evaluate(s));
  ASSERT_NO_THROW(pS->evaluateGradient(s));
  ASSERT_ANY_THROW(pS->evaluateHessian(s));
 }
} // namespace