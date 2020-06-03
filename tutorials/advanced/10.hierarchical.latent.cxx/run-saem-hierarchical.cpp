
#include "model/model.hpp"
#include "korali.hpp"

#include <vector>


ConditionalDistribution4 distrib4 = ConditionalDistribution4();

void distrib4_conditional_p(korali::Sample& s);
void distrib4_conditional_p(korali::Sample& s)
{
  distrib4.conditional_p(s);
}


int main(int argc, char* argv[])
{

  auto k = korali::Engine();
  auto e = korali::Experiment();

  int nIndividuals = distrib4._p.nIndividuals;


  e["Problem"]["Type"] = "Bayesian/Latent/HierarchicalLatent";
   // the distribution already "contains" the data, it implements the function: latent --> p(data | latent)
  e["Problem"]["Conditional Log Likelihood Function"] = &distrib4_conditional_p; // defined in model.cpp

  // We need to add one dimension to _p.data, because one individual in the general case could have
  // more than one data point assigned
  std::vector<std::vector<std::vector<double>>> dataVector(nIndividuals);
  for (size_t i = 0; i < nIndividuals; i++){
    dataVector[i].clear();
    dataVector[i].push_back(distrib4._p.data[i]);
  }
  e["Problem"]["Data"] = dataVector;
  e["Problem"][ "Data Dimensions" ] = 1;
  e["Problem"]["Number Individuals" ] = nIndividuals;
  e["Problem"][ "Latent Space Dimensions" ] = 1;

  e["Solver"]["Type"] = "HSAEM";
  e["Solver"]["Number Samples Per Step"] = 100;
  e["Solver"]["Termination Criteria"]["Max Generations"] = 30;

  e["Distributions"][0]["Name"] = "Uniform 0";
  e["Distributions"][0]["Type"] = "Univariate/Uniform";
  e["Distributions"][0]["Minimum"] = -100;
  e["Distributions"][0]["Maximum"] = 100;

    // * Define which latent variables we use (only the means - sigma is assumed known and the same for each)
  // for (size_t i = 0; i < nIndividuals; i++){
  for (size_t i = 0; i < 1; i++){
    e["Variables"][i]["Name"] = "latent mean " + std::to_string(i);
    e["Variables"][i]["Initial Value"] = -5.0;
    e["Variables"][i]["Bayesian Type"] = "Latent";
    // These are set implicitly already ('Hierarchical Latent' problem doesn't have those parameters)
//    e["Variables"][i]["Individual Index"] = i;
//    e["Variables"][i]["Latent Space Coordinate"] = 0;
    e["Variables"][i]["Latent Variable Distribution Type"] = "Normal";
    e["Variables"][i]["Prior Distribution"] = "Uniform 0" ; // not used (?) but required
  }
  e["File Output"]["Frequency"] = 50;
  e["Console Output"]["Frequency"] = 10;
  e["Console Output"]["Verbosity"] = "Detailed";


  k.run(e);

  return 0;

}
