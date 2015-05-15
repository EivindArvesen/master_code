#!/usr/bin/env bash

function message {
    echo "This script takes one arg ("Converted"), which is the folder in which it can find other folder containing .names and .files -files it will upload to the server."
    exit
}

if [[ !($1) ]]; then
    message
fi

FOLDER=$(shopt -s extglob; echo "${1%%+(/)}")

cd $FOLDER

for D in `ls -d */`
do
    cd $D
    rm -rf Results/Pylearn2/*
    python runPylearn2.py
    cd ..
done
echo "ALL RUNS ARE FINISHED"
