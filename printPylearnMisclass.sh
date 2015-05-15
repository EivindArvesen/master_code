#!/usr/bin/env bash

function message {
    echo "This script takes one arg, which is the folder in which it can find other folders containing .pkl files from cross validated Pylearn2 experiments."
    exit
}

if [[ !($1) ]]; then
    message
fi

FOLDER=$(shopt -s extglob; echo "${1%%+(/)}")

cd $FOLDER

for i in {1..4}
do
    echo "---------------------------------------- MERGE GROUP $i ----------------------------------------"
    for D in `ls -d *mergeGroups-$i`
    do
        echo "$D" | grep -oE "reduction-[a-zA-Z]*" &&  echo "$D" | grep -oE "mergeGroups-[0-4]*"
        for C in `ls -d "$D"/Results/Pylearn2/*`
        do
            printf "                " && basename $(echo "$C")
            print_monitor_cv.py "$C"/*_best.pkl | grep -E "misclass" && echo ""
        done
    done
done
