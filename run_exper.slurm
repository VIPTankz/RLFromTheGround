#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=25000
#SBATCH -p batch
#SBATCH --job-name=Vanilla
#SBATCH --time=10:00:00
#SBATCH --mail-user=tjc2g19@soton.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --account=ecsstaff
#SBATCH --partition=amd_scavenger

module load conda/py3-latest
source activate BTR

cd /home/$USER/RLFromTheGround
pip install -e .

staskfarm runs0.txt