#!/bin/bash
#SBATCH --job-name=update_pytorch     # Job name
#SBATCH --output=update_pytorch.out  # Standard output and error log
#SBATCH --error=update_pytorch.err   # Error log
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --time=01:00:00              # Time limit hrs:min:sec
#SBATCH --mem=4GB                    # Memory required per node
#SBATCH --gres=gpu:1                 # Request one GPU
#SBATCH --partition=h100              # Specify the partition (update this based on your cluster)

# Load necessary modules (if any, adjust based on your cluster)
module load cuda/11.8  # Load the CUDA module (adjust as per the cluster setup)

# Activate your virtual environment
source ~/venvGui/bin/activate

# Step 1: Check the current PyTorch version
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}')"

# Step 2: Update PyTorch to a compatible version
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 3: Verify the update
python -c "import torch; print(f'Updated PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}'); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Deactivate the virtual environment
deactivate
