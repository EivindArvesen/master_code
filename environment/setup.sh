#!/usr/bin/env bash

CWD=$(pwd)
ENV=adni
PYSRC=$HOME/src/Python

# ensure that conda is installed!

cat condarc.txt > $HOME/.condarc
cat theanorc.txt > $HOME/.theanorc

conda create --name $ENV --file requirements.txt
source activate $ENV

mkdir -p $PYSRC
cd $PYSRC
git clone git://github.com/Theano/Theano.git Theano
cd Theano
git pull
python setup.py develop
cd $PYSRC
git clone git://github.com/lisa-lab/pylearn2.git pylearn2
cd pylearn2
git pull
python setup.py develop

# THIS GOES IN ".profile" : export PYLEARN2_DATA_PATH=/data/lisa/data

cd $CWD
echo "Theano has been set up and installed to $PYSRC/Theano"
echo "Pylearn2 has been set up and installed to $PYSRC/pylearn2"
