#!/bin/bash
#SBATCH --time=0-00:05:00
# you need just a single node, kind of a master which orchestrates the jobs
#SBATCH --partition=maxgpu,psgpu,allgpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclude=max-exflg007
# that's just to ensure that all jobs run on identical nodes for benchmarking
#SBATCH --constraint=A100
#SBATCH -o /gpfs/petra3/scratch/jgruen/ci-package/ci-%j-output.log # STDOUT

ml maxwell mamba
. mamba-init
mamba activate /home/gruenjoh/jgruen/ci-package/ci_${CI_JOB_ID}_env

which python
which pytest

ls /home/gruenjoh/jgruen/ci-package/ci_${CI_JOB_ID}_env/
ls /home/gruenjoh/jgruen/ci-package/ci_${CI_JOB_ID}_env/bin/
/home/gruenjoh/jgruen/ci-package/ci_${CI_JOB_ID}_env/bin/pytest ~/jgruen/ci-package/ci_${CI_JOB_ID}/tests/
