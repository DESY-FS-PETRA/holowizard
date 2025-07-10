#!/bin/bash

resolution=$(($3 + 1))
resolution_2=$(($4 + 1))

for i in $(eval echo {0..$3}); do
  for j in $(eval echo {0..$4}); do
    sbatch  create_focus_series_multidim.sbatch RECO_FILE=$1 PYTHON_ENV=$2 RESOLUTION=$resolution RESOLUTION_2=$resolution_2 CURRENT_ID=$i CURRENT_ID_2=$j
  done
done
