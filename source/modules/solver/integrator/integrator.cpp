#include "engine.hpp"
#include "modules/solver/integrator/integrator.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
;

void Integrator::setInitialConfiguration()
{
  _variableCount = _k->_variables.size();

  _integral = 0;
  _sampleCount = 1;
  for (size_t i = 0; i < _variableCount; i++)
    _sampleCount *= _k->_variables[i]->_numberOfGridpoints;
  _maxModelEvaluations = std::min(_maxModelEvaluations, _sampleCount);

  if (_k->_variables[0]->_samplePoints.size() > 0)
  { // quadrature
    _indicesHelper.resize(_variableCount);
    _indicesHelper[0] = _k->_variables[0]->_samplePoints.size();
    _indicesHelper[1] = _k->_variables[0]->_samplePoints.size();
    for (size_t i = 2; i < _indicesHelper.size(); i++)
    {
      _indicesHelper[i] = _indicesHelper[i - 1] * _k->_variables[i - 1]->_samplePoints.size();
    }
  }
}

void Integrator::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  _executionsPerGeneration = std::min(_executionsPerGeneration, _maxModelEvaluations - _modelEvaluationCount);

  std::vector<Sample> samples(_executionsPerGeneration);
  std::vector<double> sampleData(_variableCount);
  std::vector<std::vector<size_t>> usedIndices(_executionsPerGeneration, std::vector<size_t>(_variableCount));

  size_t rest, index;
  for (size_t i = 0; i < _executionsPerGeneration; i++)
  {
    rest = _modelEvaluationCount;
    for (int d = _variableCount - 1; d >= 0; d--)
    {
      // quadrature
      //  We assume i = _index[0] + _index[1]*_sample[0].size() + _index[1]*_index[2]*_sample[1].size() + .....
      if (d == 0)
        index = rest % _indicesHelper[d];
      else
        index = rest / _indicesHelper[d];
      rest -= index * _indicesHelper[d];

      sampleData[d] = _k->_variables[d]->_samplePoints[index];
      usedIndices[i][d] = index;
    }

    _k->_logger->logInfo("Detailed", "Running sample %zu/%zu with values:\n         ", _modelEvaluationCount + 1, _sampleCount);
    for (auto &x : sampleData) _k->_logger->logData("Detailed", " %le   ", x);
    _k->_logger->logData("Detailed", "\n");

    samples[i]["Sample Id"] = i;
    samples[i]["Module"] = "Problem";
    samples[i]["Operation"] = "Execute";
    samples[i]["Parameters"] = sampleData;
    KORALI_START(samples[i]);
    _modelEvaluationCount++;
  }
  KORALI_WAITALL(samples);

  double weight;
  for (size_t i = 0; i < _executionsPerGeneration; i++)
  {
    weight = 1.0;
    for (size_t d = 0; d < _variableCount; d++)
      weight *= _k->_variables[d]->_quadratureWeights[usedIndices[i][d]];

    auto f = KORALI_GET(double, samples[i], "Evaluation");

    _integral += weight * f;
  }
  (*_k)["Results"]["Integral"] = _integral;
}

void Integrator::printGenerationBefore()
{
}

void Integrator::printGenerationAfter()
{
  _k->_logger->logInfo("Minimal", "Total Terms summed %lu/%lu.\n", _modelEvaluationCount, _sampleCount);
}

void Integrator::finalize()
{
  _k->_logger->logInfo("Minimal", "Integral Calculated: %e\n", _integral);
}

void Integrator::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Sample Count"))
 {
 try { _sampleCount = js["Sample Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ integrator ] \n + Key:    ['Sample Count']\n%s", e.what()); } 
   eraseValue(js, "Sample Count");
 }

 if (isDefined(js, "Integral"))
 {
 try { _integral = js["Integral"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ integrator ] \n + Key:    ['Integral']\n%s", e.what()); } 
   eraseValue(js, "Integral");
 }

 if (isDefined(js, "Indices Helper"))
 {
 try { _indicesHelper = js["Indices Helper"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ integrator ] \n + Key:    ['Indices Helper']\n%s", e.what()); } 
   eraseValue(js, "Indices Helper");
 }

 if (isDefined(js, "Executions Per Generation"))
 {
 try { _executionsPerGeneration = js["Executions Per Generation"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ integrator ] \n + Key:    ['Executions Per Generation']\n%s", e.what()); } 
   eraseValue(js, "Executions Per Generation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Executions Per Generation'] required by integrator.\n"); 

 Solver::setConfiguration(js);
 _type = "integrator";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: integrator: \n%s\n", js.dump(2).c_str());
} 

void Integrator::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Executions Per Generation"] = _executionsPerGeneration;
   js["Sample Count"] = _sampleCount;
   js["Integral"] = _integral;
   js["Indices Helper"] = _indicesHelper;
 Solver::getConfiguration(js);
} 

void Integrator::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Executions Per Generation\": 500000000}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void Integrator::applyVariableDefaults() 
{

 Solver::applyVariableDefaults();
} 

;

} //solver
} //korali
;
