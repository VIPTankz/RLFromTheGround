#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=25000
#SBATCH -p batch
#SBATCH --job-name=Qbert_3
#SBATCH --time=60:00:00
#SBATCH --mail-type=ALL
#SBATCH --account=ecsstaff
#SBATCH --output=worked_Qbert_3.out
#SBATCH --error=failed_Qbert_3.err
#SBATCH --partition=batch

module load conda/py3-latest
source activate BTR

cd /home/$USER/RLFromTheGround
pip install -e .

python main.py --game Qbert --agent_name NoisyNStep3DQN --fc1 128
