#include "korali.h"
#include <numeric>
#include <limits>
#include <chrono>

#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_multimin.h>

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Solver::MCMC::MCMC(nlohmann::json& js, std::string name)
{
 setConfiguration(js);

 // Allocating MCMC memory
 _covarianceMatrix  = (double*) calloc (_k->N*_k->N, sizeof(double));
 z                  = (double*) calloc (_k->N, sizeof(double));
 clPoint            = (double*) calloc (_k->N, sizeof(double));
 ccPoints           = (double*) calloc (_k->N*rejectionLevels, sizeof(double));
 transformedSamples = (double*) calloc (_k->N*rejectionLevels, sizeof(double));
 ccLogPriors        = (double*) calloc (rejectionLevels, sizeof(double));
 ccLogLikelihoods   = (double*) calloc (rejectionLevels, sizeof(double));
 alpha              = (double*) calloc (rejectionLevels, sizeof(double));
 databasePoints     = (double*) calloc (_k->N*chainLength, sizeof(double));
 databaseFitness    = (double*) calloc (chainLength, sizeof(double));
 chainMean          = (double*) calloc (_k->N, sizeof(double));
 tmpC               = (double*) calloc (_k->N*_k->N , sizeof(double));
 chainCov           = (double*) calloc (_k->N*_k->N , sizeof(double));

 for(size_t i = 0; i < _k->N; i++) clPoint[i] = variableInitialMeans[i];
 for(size_t i = 0; i < _k->N; i++) _covarianceMatrix[i*_k->N+i] = variableStandardDeviations[i];

 // Initializing Gaussian Generator
 auto jsGaussian = nlohmann::json();
 jsGaussian["Type"] = "Gaussian";
 jsGaussian["Mean"] = 0.0;
 jsGaussian["Sigma"] = 1.0;
 jsGaussian["Seed"] = _k->_seed++;
 _gaussianGenerator = new Variable();
 _gaussianGenerator->setDistribution(jsGaussian);
 
 auto jsUniform = nlohmann::json();
 jsUniform["Type"] = "Uniform";
 jsUniform["Minimum"] = 0.0;
 jsUniform["Maximum"] = 1.0;
 jsUniform["Seed"] = _k->_seed++;
 _uniformGenerator = new Variable();
 _uniformGenerator->setDistribution(jsUniform);

 // Init Generation
 _isFinished = false;
 countevals               = 0;
 naccept                  = 0;
 countgens                = 0;
 chainLength              = 0;
 rejections               = 0;
 databaseEntries          = 0;
 clLogLikelihood          = -std::numeric_limits<double>::max();
 acceptanceRateProposals  = 1.0;

 // If state is defined:
 if (isDefined(js, {"MCMC", "State"}))
 {
  setState(js);
  js["MCMC"].erase("State");
 }

}

Korali::Solver::MCMC::~MCMC()
{
 // TODO: Ensure proper memory deallocation
}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void Korali::Solver::MCMC::getConfiguration(nlohmann::json& js)
{
 js["Solver"] = "MCMC";
 
 js["MCMC"]["Result Output Frequency"] = resultOutputFrequency;

 js["MCMC"]["Chain Length"]                   = chainLength;
 js["MCMC"]["Burn In"]                        = burnIn;
 js["MCMC"]["Rejection Levels"]               = rejectionLevels;
 js["MCMC"]["Adaptive Sampling"]              = useAdaptiveSampling;
 js["MCMC"]["Non Adaption Period"]            = nonAdaptionPeriod;
 js["MCMC"]["Chain Covariance Scaling"]       = chainCovarianceScaling;
 js["MCMC"]["Chain Covariance Increment"]     = chainCovarianceIncrement;
 //js["MCMC"]["Max Resamplings"]                = _maxresamplings;

 // Variable information
 for (size_t i = 0; i < _k->N; i++)
 {
  js["Variables"][i]["MCMC"]["Initial Mean"]      = variableInitialMeans[i];
  js["Variables"][i]["MCMC"]["Initial Deviation"] = variableStandardDeviations[i];
  js["Variables"][i]["MCMC"]["Log Space"]         = variableLogSpaces[i];
 }

 js["MCMC"]["Termination Criteria"]["Max Function Evaluations"]["Value"]  = maxFunctionEvaluations;
 js["MCMC"]["Termination Criteria"]["Max Function Evaluations"]["Active"] = maxFunctionEvaluationsEnabled;

 // State Variables
 for (size_t d = 0; d < _k->N*_k->N; d++) js["MCMC"]["State"]["CovarianceMatrix"][d] = _covarianceMatrix[d];
 
 js["MCMC"]["State"]["Function Evaluations"]      = countevals;
 js["MCMC"]["State"]["Number Accepted Samples"]   = naccept;
 js["MCMC"]["State"]["Chain Length"]              = chainLength;
 js["MCMC"]["State"]["Database Entries"]          = databaseEntries;
 js["MCMC"]["State"]["Acceptance Rate Proposals"] = acceptanceRateProposals;
 js["MCMC"]["State"]["Rejections"]                = rejections;
 js["MCMC"]["State"]["Finished"]                  = _isFinished;

 for (size_t d = 0; d < _k->N; d++) js["MCMC"]["State"]["Leader"][d]                         = clPoint[d];
 for (size_t r = 0; r < rejections; ++r) for (size_t d = 0; d < _k->N; d++) js["MCMC"]["State"]["Candidates"][r][d] = ccPoints[r*_k->N+d];
 for (size_t r = 0; r < rejections; ++r) js["MCMC"]["State"]["CandidatesFitness"][r]         = ccLogLikelihoods[r];
 for (size_t i = 0; i < _k->N*databaseEntries; i++) js["MCMC"]["State"]["DatabasePoints"][i] = databasePoints[i];
 for (size_t i = 0; i < databaseEntries; i++) js["MCMC"]["State"]["DatabaseFitness"][i]      = databaseFitness[i];
 for (size_t d = 0; d < _k->N; d++) js["MCMC"]["State"]["Chain Mean"][d]                     = chainMean[d];
 for (size_t d = 0; d < _k->N * _k->N; d++) js["MCMC"]["State"]["Chain Covariance"][d]       = chainCov[d];

 js["MCMC"]["State"]["LeaderFitness"]    = clLogLikelihood;

}

void Korali::Solver::MCMC::setConfiguration(nlohmann::json& js)
{
 resultOutputFrequency    = consume(js, { "MCMC", "Result Output Frequency" }, KORALI_NUMBER, std::to_string(100));
 
 chainLength                        = consume(js, { "MCMC", "Chain Length" }, KORALI_NUMBER);
 burnIn                   = consume(js, { "MCMC", "Burn In" }, KORALI_NUMBER, std::to_string(0));
 rejectionLevels          = consume(js, { "MCMC", "Rejection Levels" }, KORALI_NUMBER, std::to_string(1));

 if (rejectionLevels < 1) { fprintf( stderr, "[Korali] MCMC Error: Rejection Level must be at least One (is %lu)\n", rejectionLevels); exit(-1); }
 
 useAdaptiveSampling      = consume(js, { "MCMC", "Use Adaptive Sampling" }, KORALI_BOOLEAN, "false");
 nonAdaptionPeriod        = consume(js, { "MCMC", "Non Adaption Period" }, KORALI_NUMBER, std::to_string(0.05 * chainLength));
 chainCovarianceScaling   = consume(js, { "MCMC", "Chain Covariance Scaling" }, KORALI_NUMBER, std::to_string(2.4*2.4/_k->N)); //Gelman et al. 1995
 chainCovarianceIncrement = consume(js, { "MCMC", "Chain Covariance Increment" }, KORALI_NUMBER, std::to_string(0.001)); // sth small (Haario et. al. 2006)
 
 if (chainCovarianceScaling < 0) { fprintf( stderr, "[Korali] MCMC Error: Chain Covariance Learning Rate must be larger Zero (is %f)\n", chainCovarianceScaling); exit(-1); }
 if (chainCovarianceIncrement < 0) { fprintf( stderr, "[Korali] MCMC Error: Chain Covariance Increment must be larger Zero (is %f)\n", chainCovarianceIncrement); exit(-1); }
 
 //_maxresamplings           = consume(js, { "MCMC", "Max Resamplings" }, KORALI_NUMBER, std::to_string(1e6));
 maxFunctionEvaluations      = consume(js, { "MCMC", "Termination Criteria", "Max Function Evaluations", "Value" }, KORALI_NUMBER, std::to_string(1e6));
 maxFunctionEvaluationsEnabled    = consume(js, { "MCMC", "Termination Criteria", "Max Function Evaluations", "Active" }, KORALI_BOOLEAN, "false");
  
 variableInitialMeans     = (double*) calloc(sizeof(double), _k->N);
 variableStandardDeviations          = (double*) calloc(sizeof(double), _k->N);
 variableLogSpaces = (bool*) calloc(sizeof(bool), _k->N);

 for(size_t d = 0; d < _k->N; d++) variableInitialMeans[d]     = consume(js["Variables"][d], { "MCMC", "Initial Mean" }, KORALI_NUMBER);
 for(size_t d = 0; d < _k->N; d++) variableStandardDeviations[d]          = consume(js["Variables"][d], { "MCMC", "Standard Deviation" }, KORALI_NUMBER);
 for(size_t d = 0; d < _k->N; d++) variableLogSpaces[d] = consume(js["Variables"][d], { "MCMC", "Log Space" }, KORALI_BOOLEAN, "false");
  
 for(size_t d = 0; d < _k->N; d++) if (variableStandardDeviations[d] < 0) { fprintf( stderr, "[Korali] MCMC Error: Initial Standard Deviation in dim %zu must be larger Zero (is %f)\n", d, variableStandardDeviations[d]); exit(-1); }

}

void Korali::Solver::MCMC::setState(nlohmann::json& js)
{
 countevals              = js["MCMC"]["State"]["FunctionEvaluations"];
 naccept                 = js["MCMC"]["State"]["Number Accepted Samples"];
 chainLength             = js["MCMC"]["State"]["Chain Length"];
 databaseEntries         = js["MCMC"]["State"]["Database Entries"];
 acceptanceRateProposals = js["MCMC"]["State"]["Acceptance Rate Proposals"];
 rejections              = js["MCMC"]["State"]["Rejections"];
 _isFinished             = js["MCMC"]["State"]["Finished"];

 for (size_t d = 0; d < _k->N; d++) clPoint[d]                        = js["MCMC"]["State"]["Leader"][d];
 for (size_t r = 0; r < rejections; ++r) for (size_t d = 0; d < _k->N; d++) ccPoints[r*_k->N+d] = js["MCMC"]["State"]["Candidate"][r][d];
 for (size_t r = 0; r < rejections; ++r) ccLogLikelihoods[r]          = js["MCMC"]["State"]["CandidatesFitness"][r];
 for (size_t i = 0; i < _k->N*databaseEntries; i++) databasePoints[i] = js["MCMC"]["State"]["DatabasePoints"][i];
 for (size_t i = 0; i < databaseEntries; i++) databaseFitness[i]      = js["MCMC"]["State"]["DatabaseFitness"][i];
 for (size_t d = 0; d < _k->N; d++) chainMean[d]                      = js["MCMC"]["State"]["Chain Mean"][d];
 for (size_t d = 0; d < _k->N * _k->N; d++) chainCov[d]               = js["MCMC"]["State"]["Chain Covariance"][d];
 for (size_t d = 0; d < _k->N*_k->N; d++) _covarianceMatrix[d]        = js["MCMC"]["State"]["CovarianceMatrix"][d];
 
 clLogLikelihood = js["MCMC"]["State"]["LeaderFitness"];

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Solver::MCMC::run()
{
 saveState();
 if (_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Starting MCMC.\n");
 
 startTime = std::chrono::system_clock::now();

 while(!checkTermination())
 {
  t0 = std::chrono::system_clock::now();

  rejections = 0;
  while( rejections < rejectionLevels )
  {
    generateCandidate(rejections);
    evaluateSample();
    _k->_conduit->checkProgress();
    acceptReject(rejections);
    rejections++;
  }
  chainLength++;
  if (chainLength > burnIn ) updateDatabase(clPoint, clLogLikelihood);
  updateState();

  t1 = std::chrono::system_clock::now();

  saveState();
  
  printGeneration();
 }

 printFinal();

 endTime = std::chrono::system_clock::now();
}


void Korali::Solver::MCMC::processSample(size_t sampleIdx, double fitness)
{
 ccLogLikelihoods[sampleIdx] = fitness + ccLogPriors[sampleIdx];
}


void Korali::Solver::MCMC::acceptReject(size_t trial)
{

 double denom;
 double alpha = recursiveAlpha(denom, clLogLikelihood, ccLogLikelihoods, trial);
 if ( alpha == 1.0 || alpha > _uniformGenerator->getRandomNumber() ) {
   naccept++;
   clLogLikelihood = ccLogLikelihoods[trial];
   for (size_t d = 0; d < _k->N; d++) clPoint[d] = ccPoints[trial*_k->N+d];
 }
}


double Korali::Solver::MCMC::recursiveAlpha(double& denom, const double llk0, const double* logliks, size_t N) const
{
    // recursive formula from Trias[2009]

    if(N==0)
    {
        denom = exp(llk0);
        return std::min(1.0, exp(logliks[0] - llk0));
    }
    else
    {
        // revert sample array
        double* revLlks = new double[N];
        for(size_t i = 0; i < N; ++i) revLlks[i] = logliks[N-1-i];
        
        // update numerator (w. recursive calls)
        double numerator = std::exp(logliks[N]);
        for(size_t i = 0; i < N; ++i)
        {   
            double denom2; 
            double recalpha2 = recursiveAlpha(denom2, logliks[N], revLlks, i);
            numerator *=  ( 1.0 - recalpha2 );
        }
        delete [] revLlks;
  
        if (numerator == 0.0) return 0.0;

        // update denomiator
        double denom1;
        double recalpha1 = recursiveAlpha(denom1, llk0, logliks, N-1);
        denom = denom1 * (1.0 - recalpha1);
               
        return std::min(1.0, numerator/denom);
    }
}


void Korali::Solver::MCMC::updateDatabase(double* point, double loglik)
{
 for (size_t d = 0; d < _k->N; d++) databasePoints[databaseEntries*_k->N + d] = point[d];
 databaseFitness[databaseEntries] = loglik;
 databaseEntries++;
}


void Korali::Solver::MCMC::generateCandidate(size_t sampleIdx)
{  
 size_t initialgens = countgens;
 sampleCandidate(sampleIdx); countgens++;
 setCandidatePriorAndCheck(sampleIdx);
 
 /*do { /TODO: fix check (DW)
   sampleCandidate(level); countgens++;
   setCandidatePriorAndCheck(level);
   if ( (countgens - initialgens) > _maxresamplings) 
   {       
     if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: exiting resampling loop, max resamplings (%zu) reached.\n", _maxresamplings);
     exit(-1);
   }
  } while (setCandidatePriorAndCheck(level)); */
}

void Korali::Solver::MCMC::evaluateSample()
{
  for(size_t d = 0; d < _k->N; ++d)
    if(variableLogSpaces[d] == true)
        transformedSamples[rejections*_k->N+d] = std::exp(ccPoints[rejections*_k->N+d]);
    else 
        transformedSamples[rejections*_k->N+d] = ccPoints[rejections*_k->N+d];

  _k->_conduit->evaluateSample(transformedSamples, rejections); countevals++;
}

void Korali::Solver::MCMC::sampleCandidate(size_t sampleIdx)
{  
 for (size_t d = 0; d < _k->N; ++d) { z[d] = _gaussianGenerator->getRandomNumber(); ccPoints[sampleIdx*_k->N+d] = 0.0; }

 if ( (useAdaptiveSampling == false) || (databaseEntries <= nonAdaptionPeriod + burnIn))
     for (size_t d = 0; d < _k->N; ++d) for (size_t e = 0; e < _k->N; ++e) ccPoints[sampleIdx*_k->N+d] += _covarianceMatrix[d*_k->N+e] * z[e];
 else
     for (size_t d = 0; d < _k->N; ++d) for (size_t e = 0; e < _k->N; ++e) ccPoints[sampleIdx*_k->N+d] += chainCov[d*_k->N+e] * z[e];

 for (size_t d = 0; d < _k->N; ++d) ccPoints[sampleIdx*_k->N+d] += clPoint[d];
}


bool Korali::Solver::MCMC::setCandidatePriorAndCheck(size_t sampleIdx)
{
 ccLogPriors[sampleIdx] = _k->_problem->evaluateLogPrior(ccPoints+_k->N*sampleIdx);
 if (ccLogPriors[sampleIdx] > -INFINITY) return true;
 return false;
}


void Korali::Solver::MCMC::updateState()
{

 acceptanceRateProposals = ( (double)naccept/ (double)chainLength );

 if(databaseEntries < 1) return;
 
 for (size_t d = 0; d < _k->N; d++) for (size_t e = 0; e < d; e++)
 {
   tmpC[d*_k->N+e] = databaseEntries*chainMean[d]*chainMean[e] + clPoint[d]*clPoint[e];
   tmpC[e*_k->N+d] = databaseEntries*chainMean[d]*chainMean[e] + clPoint[d]*clPoint[e];
 }
 for (size_t d = 0; d < _k->N; d++) tmpC[d*_k->N+d] = databaseEntries*chainMean[d]*chainMean[d] + clPoint[d]*clPoint[d] + chainCovarianceIncrement;

 // Chain Mean
 for (size_t d = 0; d < _k->N; d++) chainMean[d] = ((chainMean[d] * (databaseEntries-1) + clPoint[d])) / ((double) databaseEntries);
 
 for (size_t d = 0; d < _k->N; d++) for (size_t e = 0; e < d; e++)
 {
    tmpC[d*_k->N+e] -= (databaseEntries+1)*chainMean[d]*chainMean[e];
    tmpC[e*_k->N+d] -= (databaseEntries+1)*chainMean[d]*chainMean[e];

 }
 for (size_t d = 0; d < _k->N; d++) tmpC[d*_k->N+d] -= (databaseEntries+1)*chainMean[d]*chainMean[d];

 // Chain Covariance (TODO: careful check N (databasEntires) (DW)
 for (size_t d = 0; d < _k->N; d++) for (size_t e = 0; e < d; e++)
 {
   chainCov[d*_k->N+e] = (databaseEntries-1.0)/( (double) databaseEntries) * chainCov[d*_k->N+e] + (chainCovarianceScaling/( (double) databaseEntries))*tmpC[d*_k->N+e];
   chainCov[e*_k->N+d] = (databaseEntries-1.0)/( (double) databaseEntries) * chainCov[d*_k->N+e] + (chainCovarianceScaling/( (double) databaseEntries))*tmpC[d*_k->N+e];
 }
 for (size_t d = 0; d < _k->N; d++) chainCov[d*_k->N+d] = (databaseEntries-1.0)/( (double) databaseEntries) * chainCov[d*_k->N+d] + (chainCovarianceScaling/( (double) databaseEntries))*tmpC[d*_k->N+d];

 
}

bool Korali::Solver::MCMC::checkTermination()
{

 if ( maxFunctionEvaluationsEnabled && (countevals >= maxFunctionEvaluations))
 {
  _isFinished = true;
  sprintf(_terminationReason, "Max Function Evaluations reached (%zu)",  countevals);
 }
 
 if ( databaseEntries == chainLength)
 {
  _isFinished = true;
  sprintf(_terminationReason, "Chainlength (%zu) reached.",  chainLength);
 }

 return _isFinished;
}
 

void Korali::Solver::MCMC::saveState() const
{
 if (_isFinished || (chainLength % resultOutputFrequency) == 0) _k->saveState(chainLength);
}

 
void Korali::Solver::MCMC::printGeneration() const
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
  printf("--------------------------------------------------------------------\n");
  printf("[Korali] Database Entries %ld\n", databaseEntries);
  printf("[Korali] Duration: %fs (Elapsed Time: %.2fs)\n",  
          std::chrono::duration<double>(t1-t0).count() , 
          std::chrono::duration<double>(t1-startTime).count());
 }

 if (_k->_verbosity >= KORALI_NORMAL) printf("[Korali] Accepted Samples: %zu\n", naccept);
 if (_k->_verbosity >= KORALI_NORMAL) printf("[Korali] Acceptance Rate Proposals: %.2f%%\n", 100*acceptanceRateProposals);

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Variable = (Current Sample, Current Candidate):\n");
  for (size_t d = 0; d < _k->N; d++)  printf("         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), clPoint[d], ccPoints[d]);
  printf("[Korali] Current Chain Mean:\n");
  for (size_t d = 0; d < _k->N; d++) printf(" %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), chainMean[d]);
  printf("[Korali] Current Chain Covariance:\n");
  for (size_t d = 0; d < _k->N; d++)
  {
   for (size_t e = 0; e <= d; e++) printf("   %+6.3e  ", chainCov[d*_k->N+e]);
   printf("\n");
  }


 }
}


void Korali::Solver::MCMC::printFinal() const
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("[Korali] MCMC Finished\n");
    printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
    printf("[Korali] Number of Generated Samples: %zu\n", countgens);
    printf("[Korali] Acceptance Rate: %.2f%%\n", 100*acceptanceRateProposals);
    if (databaseEntries == chainLength) printf("[Korali] Max Samples Reached.\n");
    else printf("[Korali] Stopping Criterium: %s\n", _terminationReason);
    printf("[Korali] Total Elapsed Time: %fs\n", std::chrono::duration<double>(t1-startTime).count());
    printf("--------------------------------------------------------------------\n");
 }
}

