#! /usr/bin/env python3
import os
import sys
import signal
import json
import argparse
import time
import matplotlib
import importlib
import math 
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from korali.plot.helpers import hlsColors, drawMulticoloredLine
from scipy.signal import savgol_filter

# Check if name has correct suffix
def validateOutput(output):
  if not (output.endswith(".png") or output.endswith(".eps") or output.endswith(".svg")):
    print("[Korali] Error: Outputfile '{0}' must end with '.eps', '.png' or '.svg' suffix.".format(output))
    sys.exit(-1)

##################### Plotting Reward History

def plotRewardHistory(ax, dirs, results, minReward, maxReward, averageDepth, maxObservations, showCI, aggregate):

 ## Setting initial x-axis (episode) and  y-axis (reward) limits
 
 maxPlotObservations = -math.inf
 maxPlotReward = -math.inf
 minPlotReward = +math.inf

 ## Creating colormap
 cmap = matplotlib.cm.get_cmap('brg')
 colCurrIndex = 0.0

 ## Reading the individual results

 unpackedResults = []
 for r in results:
  
  if (len(r) == 0): continue  
  
  cumulativeObsCountHistory = np.cumsum(np.array(r["Solver"]["Training"]["Experience History"]))
  rewardHistory = np.array(r["Solver"]["Training"]["Reward History"])
  trainingRewardThreshold = r["Problem"]["Training Reward Threshold"]
  testingRewardThreshold = r["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"]

  # Merge Results
  if aggregate == True and len(unpackedResults) > 0:
    coH, rH, trTh, teTh = unpackedResults[0]
    aggCumObs = np.append(coH, cumulativeObsCountHistory)
    aggRewards = np.append(rH, rewardHistory)

    sortedAggRewards = np.array([r for _, r in sorted(zip(aggCumObs, aggRewards), key=lambda pair: pair[0])])
    sortedAggCumObs = np.sort(aggCumObs)
    unpackedResults[0] = (sortedAggCumObs, sortedAggRewards, trainingRewardThreshold, testingRewardThreshold)

  # Append Results
  else:
    unpackedResults.append( (cumulativeObsCountHistory, rewardHistory, trainingRewardThreshold, testingRewardThreshold) )

 ## Plotting the individual experiment results
    
 for resId, r in enumerate(unpackedResults):
  
  cumulativeObsArr, rewardHistory, trainingRewardThreshold, testingRewardThreshold = r
  
  currObsCount = cumulativeObsArr[-1]
  
  # Updating common plot limits
 
  if (currObsCount > maxPlotObservations): maxPlotObservations = currObsCount
  if (maxObservations): maxPlotObservations = int(maxObservations)

  if (min(rewardHistory) < minPlotReward): 
   if (min(rewardHistory) > -math.inf):
    minPlotReward = min(rewardHistory)
  
  if (max(rewardHistory) > maxPlotReward):
   if (max(rewardHistory) < math.inf):
    maxPlotReward = max(rewardHistory)

  if (trainingRewardThreshold != -math.inf and trainingRewardThreshold != math.inf): 
   if (trainingRewardThreshold > maxPlotReward): maxPlotReward = trainingRewardThreshold

  if (testingRewardThreshold != -math.inf and testingRewardThreshold != math.inf): 
   if (testingRewardThreshold > maxPlotReward): maxPlotReward = testingRewardThreshold
  
  if (trainingRewardThreshold != -math.inf and trainingRewardThreshold != math.inf):   
   if (trainingRewardThreshold < minPlotReward): minPlotReward = trainingRewardThreshold
   
  if (testingRewardThreshold != -math.inf and testingRewardThreshold != math.inf): 
   if (testingRewardThreshold < minPlotReward): minPlotReward = testingRewardThreshold
 
  # Getting average cumulative reward statistics
  cumRewards = np.cumsum(rewardHistory)
  meanHistoryStart = cumRewards[:averageDepth]/np.arange(1,averageDepth+1)
  meanHistoryEnd = (cumRewards[averageDepth:]-cumRewards[:-averageDepth])/float(averageDepth)
  meanHistory = np.append(meanHistoryStart, meanHistoryEnd)

  confIntervalLowerHistory = None
  confIntervalUpperHistory = None

  # Calculating confidence intervals
  if showCI > 0.0:
    confIntervalLowerHistory= [ rewardHistory[0] ]
    confIntervalUpperHistory= [ rewardHistory[0] ]

    for i in range(1, len(rewardHistory)):
      startPos = max(i - averageDepth, 0)
      endPos = i
      data = rewardHistory[startPos:endPos]
      ciLow = np.percentile(data, 50-50*showCI)
      ciUp = np.percentile(data, 50+50*showCI)
      confIntervalLowerHistory.append(ciLow)
      confIntervalUpperHistory.append(ciUp)
      
    confIntervalLowerHistory = np.array(confIntervalLowerHistory)
    confIntervalUpperHistory = np.array(confIntervalUpperHistory)

  # Plotting common plot
  ax.plot(cumulativeObsArr, rewardHistory, 'x', markersize=1.3, color=cmap(colCurrIndex), alpha=0.15, zorder=0)
  ax.plot(cumulativeObsArr, meanHistory, '-', color=cmap(colCurrIndex), lineWidth=3.0, zorder=1) 

  # Plotting confidence intervals
  if showCI > 0.:
    ax.fill_between(cumulativeObsArr, confIntervalLowerHistory, confIntervalUpperHistory, color=cmap(colCurrIndex), alpha=0.2)

  # Updating color index
  if (len(results) > 1):
   colCurrIndex = colCurrIndex + (1.0 / float(len(results)-1)) - 0.0001
  
 ## Configuring common plotting features

 if (minReward): minPlotReward = float(minReward)
 if (maxReward): maxPlotReward = float(maxReward)
 
 ax.set_ylabel('Cumulative Reward')  
 ax.set_xlabel('# Observations')
 ax.set_title('Korali RL History Viewer')
 
 ax.yaxis.grid()
 ax.set_xlim([0, maxPlotObservations-1])
 ax.set_ylim([minPlotReward - 0.1*abs(minPlotReward), maxPlotReward + 0.1*abs(maxPlotReward)])
 
##################### Results parser

def parseResults(dir):

 results = [ ]

 for p in dir:
  configFile = p + '/latest'
  if (not os.path.isfile(configFile)):
   print("[Korali] Error: Did not find any results in the {0} folder...".format(p))
   exit(-1)
 
  with open(configFile) as f:
    data = json.load(f)

  results.append(data)
  
 return results
 
##################### Main Routine: Parsing arguments and result files
  
if __name__ == '__main__':
 
 # Setting termination signal handler
 
 signal.signal(signal.SIGINT, lambda x, y: exit(0))

 # Parsing arguments

 parser = argparse.ArgumentParser(
     prog='korali.rlview',
     description='Plot the results of a Korali Reinforcement Learning execution.')
 parser.add_argument(
     '--dir',
     help='Path(s) to result files, separated by space',
     default=['_korali_result'],
     required=False,
     nargs='+')
 parser.add_argument(
     '--maxObservations',
     help='Maximum observations (x-axis) to display',
     type=int,
     default=None,
     required=False)
 parser.add_argument(
     '--maxReward',
     help='Maximum reward to display',
     default=None,
     required=False)
 parser.add_argument(
     '--updateFrequency',
     help='Specified the time (seconds) between live updates to the plot',
     default=0.0,
     required=False)
 parser.add_argument(
     '--maxPlottingTime',
     help='Specified the maximum time (seconds) to update the plot for (for testing purposes)',
     default=0.0,
     required=False)
 parser.add_argument(
     '--minReward',
     help='Minimum reward to display',
     default=None,
     required=False)
 parser.add_argument(
      '--averageDepth',
      help='Specifies the depth for plotting average',
      type=int,
      default=100,
      required=False)
 parser.add_argument(
      '--showCI',
      help='Option to plot the reward confidence interval.',
      type = float,
      default=0.0,
      required=False)
 parser.add_argument(
      '--aggregate',
      help='Aggregate multiple runs and plot result summary.',
      action='store_true')
 parser.add_argument(
      '--test',
      help='Run without graphics (for testing purpose)',
      action='store_true',
      required=False)
 parser.add_argument(
      '--output',
      help='Indicates the output file path. If not specified, it prints to screen.',
      required=False)

 args = parser.parse_args()

 ### Validating input

 if args.showCI < 0.0 or args.showCI > 1.0:
  print("[Korali] Argument of confidence interval must be in [0,1].")
  exit(-1)

 if args.output:
    validateOutput(args.output)

 ### Setup without graphics, if needed
 
 if (args.test or args.output): 
     matplotlib.use('Agg')
 
 ### Reading values from result files

 results = parseResults(args.dir)
 if (len(results) == 0): 
  print('Error: No result folders have been provided for plotting.')
  exit(-1)
 
 ### Creating figure(s)
  
 fig1 = plt.figure()
 ax1 = fig1.add_subplot(111)
     
 ### Creating plots
  
 plotRewardHistory(ax1, args.dir, results, args.minReward, args.maxReward, args.averageDepth, args.maxObservations, args.showCI, args.aggregate)
 plt.draw()
 
 ### Printing live results if update frequency > 0
 
 fq = float(args.updateFrequency)
 maxTime = float(args.maxPlottingTime)
 
 if (fq > 0.0):
  initialTime = time.time()
  while(True):
   results = parseResults(args.dir)
   plt.pause(fq)
   ax1.clear()
   plotRewardHistory(ax1, args.dir, results, args.minReward, args.maxReward, args.averageDepth, args.maxObservations, args.showCI, args.aggregate)
   plt.draw()
   
   # Check if maximum time exceeded
   if (maxTime> 0.0):
    currentTime = time.time()
    elapsedTime = currentTime - initialTime
    if (elapsedTime > maxTime):
     print('[Korali] Maximum plotting time reached. Exiting...') 
     exit(0)
 

   plt.show()
  exit(0)

 if (args.output is None):
   plt.show()
 else:
   if args.output.endswith('.eps'):
     plt.savefig(args.output, format='eps')
   elif args.output.endswith('.svg'):
     plt.savefig(args.output, format='svg')
   else:
     plt.savefig(args.output, format='png')
