#!/usr/bin/env bash

var=0
for D in `ls -d */`
do
    cd $D
    bash runC5.sh > MASTER.out 2>&1 &
    cd ..
    ((var++))
done
echo "Started (several) C5.0 runs for $var variations of the dataset."
