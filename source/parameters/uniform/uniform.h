#ifndef _KORALI_UNIFORM_H_
#define _KORALI_UNIFORM_H_

#include "parameters/base/base.h"

namespace Korali::Parameter
{

class Uniform : public Korali::Parameter::Base
{
 private:

 public:

  double getDensity(double x);
  double getDensityLog(double x);
  double getRandomNumber();

  // Serialization Methods
  nlohmann::json getConfiguration();
  void setConfiguration(nlohmann::json& js);
};

} // namespace Korali

#endif // _KORALI_DISTRIBUTION_H_
