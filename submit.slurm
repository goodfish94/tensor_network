#!/bin/sh
#SBATCH --job-name=dmft_mps
#SBATCH --partition=commons
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --mail-user=goodfish94@outlook.com
#SBATCH --export=ALL
python /projects/qmsi/hh25/projects/qmsi/hh25/dmft_mps/code/main
