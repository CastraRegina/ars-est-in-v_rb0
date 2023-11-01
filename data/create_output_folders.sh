#!/bin/bash

FOLDERS="output/example output/example/png output/example/png/checkerboard output/example/svg"

for folder in ${FOLDERS} ; do
  echo mkdir -p ${folder}
  mkdir -p ${folder}
done



