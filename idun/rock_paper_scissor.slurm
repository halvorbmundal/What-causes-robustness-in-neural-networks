#!/bin/sh
#SBATCH --partition=GPUQ,V100-IDI
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=27
#SBATCH --mem=600000
# Memory per node specification is in MB. It is optional.
# The default limit is 3000MB per core.
#SBATCH --job-name="gpu-r-CNN-Cert"
#SBATCH --output=out/10rock_output.out
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "We are using $SLURM_GPUS_ON_NODE GPUs per node"
echo "Total of $SLURM_NTASKS cores"

echo "CUDA_VISILE DEVICES: $CUDA_VISIBLE_DEVICES"

# Load required modules
module purge
# The GNU Compiler Collection includes front ends for C, C++, Objective-C, Fortran, Java, and Ada, as well as libraries for
# these languages (libstdc++, libgcj,...). - Homepage: http://gcc.gnu.org/
module load GCC/8.2.0-2.31.1

# CUDA (formerly Compute Unified Device Architecture) is a parallel computing platform and programming model created by NVIDIA
# and implemented by the graphics processing units (GPUs) that they produce. CUDA gives developers access to the virtual
# instruction set and memory of the parallel computational elements in CUDA GPUs.
module load CUDA/10.1.105
module load OpenMPI/3.1.3

echo "loading python"
module load Python/3.7.2

# module load goolfc/2017b
module load TensorFlow/1.13.1-Python-3.7.2

# Create environment if it doesn't exist
if [ ! -d "gpu_env" ]; then
  python3 -m venv gpu_env
fi

# Activate enviroment and install required dependencies
source gpu_env/bin/activate
pip install -r cnn-cert-master/idun/requirements.txt

# List information about available GPUs
nvidia-smi

# Start training
python cnn-cert-master/cnn_robustness_tester.py cpu gpu no_debugging v10 rockpaperscissors upper 27 no_adv

# Print information about the system
uname -a
