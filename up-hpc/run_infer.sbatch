#!/bin/bash
#SBATCH --job-name=snn_infer          # Job name
#SBATCH --output=logs/infer_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=logs/infer_%j.err     # Standard error log
#SBATCH --time=02:00:00               # Time limit (2 hours)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=4             # CPU cores per task
#SBATCH --mem=16G                     # Memory per node
#SBATCH --partition=gpu               # Partition name (change to your cluster's GPU partition)
#SBATCH --gres=gpu:1                  # Request 1 GPU (remove if running on CPU only)

# Optional: Email notifications
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your.email@domain.com

echo "=========================================="
echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "=========================================="

# Load modules (adjust based on your cluster)
# Example: module load cuda/11.8 python/3.10
# Uncomment and modify the line below based on your cluster's available modules
# module load python/3.10 cuda/11.8

# Activate your conda environment
# Option 1: If using conda
source activate torch310  # Replace 'torch310' with your environment name

# Option 2: If using virtualenv
# source ~/venv/bin/activate

# Print environment info
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo "CUDA available:" 
python -c "import torch; print(torch.cuda.is_available())"
echo ""

# Navigate to your project directory
cd $SLURM_SUBMIT_DIR  # This is where you submitted the job from
echo "Working directory: $(pwd)"
echo ""

# Run the inference script
echo "Starting inference..."
python infer.py

echo ""
echo "=========================================="
echo "Job finished on $(date)"
echo "=========================================="
