#!/bin/bash

function check()
{
 if [ ! $? -eq 0 ]
 then
  echo "[Korali] Error building site."
  exit -1
 fi 
}


# Installing shpinx, mkdocs, and materials theme
python3 -m pip install sphinx --user
check

python3 -m pip install sphinx_rtd_theme --user
check

# Building User Manual
pushd manual

make html
check

popd

# Inserting user manual into website

mkdir -p web/docs
check 

cp -r manual/.build/html/* web/docs
check

echo "[Korali] Webpage Build complete."
