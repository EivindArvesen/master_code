#!/usr/bin/env bash


if [[ -z "$1" ]]; then
    echo "Please enter one argument (folder containing .xml and .nii files.)"
    exit 1
fi

CONVERTER=adniConverter.py

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    python $CONVERTER $1
    exit 0
fi

FOLDER=$(shopt -s extglob; echo "${1%%+(/)}")

declare -a batches=(
    "-f C V -i 6 -m 2 -n 4 -r P -s 192 192 160 -x 1 2 3 4"
    "-f C V -i 6 -m 2 -n 1 -r H -s 192 192 160 -x 1 2 3 4"
    "-f C V -i 6 -m 2 -n 1 -s 68 95 79 -x 1 2 3 4"
)

for BATCH in "${batches[@]}";
do
    python $CONVERTER -c -d $FOLDER $BATCH
done
