#!/bin/bash
#SBATCH --time=0-00:05:00
# you need just a single node, kind of a master which orchestrates the jobs
#SBATCH --partition=maxgpu,psgpu,allgpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclude=max-exflg007
# that's just to ensure that all jobs run on identical nodes for benchmarking
#SBATCH --constraint=A100|V100|P100
#SBATCH -o ./slurm_output/output.%a.out # STDOUT

for var in "$@";do export $var; done

# the basic setup
export LD_PRELOAD=""

source /etc/profile.d/modules.sh
module load maxwell

module load mamba
#mamba activate ${PYTHON_ENV}

#source activate base
source activate ${PYTHON_ENV}

python ${RECO_FILE} ${RESOLUTION} ${CURRENT_ID} ${RESOLUTION_2} ${CURRENT_ID_2}
