#ifndef _UTILS_HPP_
#define _UTILS_HPP_


#include <cmath>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>


double multivariate_gaussian_probability(std::vector<std::vector<double> > mus,
                                         int nDimensions,
                                         std::vector<int> assignments,
                                         int nClusters,
                                         double sigma,
                                         std::vector<std::vector<double> > points);

double univariate_gaussian_probability(std::vector<double> mu,
                                       double sigma,
                                       std::vector<double> point);


#endif
