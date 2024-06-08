#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=25000
#SBATCH -p batch
#SBATCH --job-name=NStep3DQN_game_doubledunk
#SBATCH --time=40:00:00
#SBATCH --mail-type=ALL
#SBATCH --account=ecsstaff
#SBATCH --output=worked_doubledunk.out
#SBATCH --error=failed_doubledunk.err
#SBATCH --partition=batch

module load conda/py3-latest
source activate BTR

cd /home/$USER/RLFromTheGround
pip install -e .

python main.py --game DoubleDunk
