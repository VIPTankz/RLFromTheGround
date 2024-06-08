#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=8
#SBATCH --mem=25000
#SBATCH -p batch
#SBATCH --job-name=Vanilla
#SBATCH --time=40:00:00
#SBATCH --mail-type=ALL
#SBATCH --account=ecsstaff
#SBATCH --partition=amd_scavenger
#SBATCH --partition=batch

module load conda/py3-latest
source activate BTR

cd /home/$USER/RLFromTheGround
pip install -e .

staskfarm runs0.txt
