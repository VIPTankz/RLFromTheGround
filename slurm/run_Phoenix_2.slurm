#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=25000
#SBATCH -p batch
#SBATCH --job-name=Phoenix_2
#SBATCH --time=60:00:00
#SBATCH --mail-type=ALL
#SBATCH --account=ecsstaff
#SBATCH --output=worked_Phoenix_2.out
#SBATCH --error=failed_Phoenix_2.err
#SBATCH --partition=batch

module load conda/py3-latest
source activate BTR

cd /home/$USER/RLFromTheGround
pip install -e .

python main.py --game Phoenix --agent_name NStep3DQN
