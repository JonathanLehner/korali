#ifndef _KORALI_GAMMA_H_
#define _KORALI_GAMMA_H_

#include "parameters/base/base.h"

namespace Korali::Parameter
{

class Gamma : public Korali::Parameter::Base
{
 private:
  double _shape;
  double _rate;
  double _aux;

 public:

  Gamma();
  Gamma(double shape, double rate);
  Gamma(double shape, double rate, size_t seed);

  double getDensity(double x);
  double getDensityLog(double x);
  double getRandomNumber();

  void printDetails() override;

  // Serialization Methods
  nlohmann::json getConfiguration();
  void setConfiguration(nlohmann::json js);
};

} // namespace Korali

#endif // _KORALI_GAMMA_H_
