#!/usr/bin/env bash

# to run in bg: ./runC5.sh > MASTER.out 2>&1 &

STEM="ADNI"
FOLDER="Results/C5.0"
mkdir -p $FOLDER

c5.0 -f $STEM -e -S 70 > $FOLDER/adni-e-S70.log &&
c5.0 -f $STEM -e -S 70 -r > $FOLDER/adni-e-S70-r.log &&
c5.0 -f $STEM -e -S 70 -b > $FOLDER/adni-e-S70-b.log &&
c5.0 -f $STEM -e -S 70 -b -r > $FOLDER/adni-e-S70-b-r.log &&
c5.0 -f $STEM -e -S 70 -b -c 80 > $FOLDER/adni-e-S70-b-c80.log &&
c5.0 -f $STEM -e -S 70 -b -c 80 -r > $FOLDER/adni-e-S70-b-c80-r.log &&
c5.0 -f $STEM -e -S 70 -b -t 20 -g > $FOLDER/adni-e-S70-b-t20-g.log &&
c5.0 -f $STEM -e -S 70 -b -t 20 -g -r > $FOLDER/adni-e-S70-b-t20-g-r.log &&
c5.0 -f $STEM -e -S 70 -b -t 20 -g -w > $FOLDER/adni-e-S70-b-t20-g-w.log &&
c5.0 -f $STEM -e -S 70 -b -t 20 -g -w -r > $FOLDER/adni-e-S70-b-t20-g-w-r.log &&
c5.0 -f $STEM -e -S 70 -b -t 20 -c 80 -w > $FOLDER/adni-e-S70-b-t20-c80-w.log &&
c5.0 -f $STEM -e -S 70 -b -t 20 -c 80 -w -r > $FOLDER/adni-e-S70-b-t20-c80-w-r.log &&
touch RUN_IS_FINISHED.tmp
