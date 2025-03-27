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
module load gcc cuda cudnn python
source venvGui_kuma/bin/activate
srun RunTestPts.csh $1

echo "++++++++++ done!"

exit 0
