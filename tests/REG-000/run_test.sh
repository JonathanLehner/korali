#!/bin/bash

function check_result()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error running test. Please check $logfile."
  exit -1
 fi 
}

#Running tutorials

curdir=$PWD
logfile=$curdir/test.log

echo "[Korali] Beginning python tests" > $logfile
