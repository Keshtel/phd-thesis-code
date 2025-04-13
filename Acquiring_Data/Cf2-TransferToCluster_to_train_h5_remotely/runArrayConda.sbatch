#!/bin/bash
#SBATCH -D ./
#SBATCH --mail-user=mahsa.barzegarkeshteli@epfl.ch                             
#SBATCH --mail-type=END 
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=1
#SBATCH --time=47:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --mem=180G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
echo "++++++++++ resources granted"
module load gcc cuda cudnn
# Load Conda
eval "$(/home/barzegar/miniconda3/bin/conda shell.bash hook)"

# Activate the Conda environment
conda activate /scratch/barzegar/conda_envs/Conda_Gui

srun RunTest2-partial.csh SecLayer0920_$1_$2_$SLURM_ARRAY_TASK_ID $1 $2 $3

echo "++++++++++ done!"

exit 0
