#!/bin/bash

resolution=$(($2 + 1))

files=("magnesium_wire.py" "spider_hair.py")

for file in ${files[@]}; do
for i in $(eval echo {0..$2}); do
  sbatch create_focus_series_singledim.sbatch RECO_FILE=$file PYTHON_ENV=$1 RESOLUTION=$resolution CURRENT_ID=$i
done
done