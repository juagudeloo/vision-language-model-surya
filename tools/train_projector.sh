#!/bin/bash -l
################################################################################
# Projector training launcher (fisica cluster, node maxwell, one GPU)
#
# Usage:
#   mkdir -p logs                        # SLURM does not create the output folder, and logs/ is gitignored
#   sbatch tools/train_projector.sh [configs/prototype.yaml]
#   bash tools/train_projector.sh [configs/prototype.yaml]    # run directly, e.g. a quick check
#
# The config path is relative to the repo root. Everything about the run lives in that file.
################################################################################

#SBATCH --job-name=vlm_proj
#SBATCH --cluster=fisica
#SBATCH -wmaxwell
#SBATCH --partition=gpu.cecc            # the partition that contains maxwell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                       # CPU RAM while the 16 GB of bf16 weights are read from disk
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratchsan/observatorio/juagudeloo/Doctorado/vision-language-model-surya/logs/train_%j.out
#SBATCH --error=/scratchsan/observatorio/juagudeloo/Doctorado/vision-language-model-surya/logs/train_%j.err

PROJECT_ROOT="/scratchsan/observatorio/juagudeloo/Doctorado/vision-language-model-surya"
CONFIG="${1:-configs/prototype.yaml}"
cd "${PROJECT_ROOT}" || exit 1

# conda is sourced from its real install instead of `module load envs/anaconda3`: that module puts
# /scratch1/anaconda3 on PATH, which does not exist on maxwell, and a login shell here never reads
# ~/.bashrc, so nothing else would define `conda`.
CONDA_SH="/scratchsan1/anaconda3/etc/profile.d/conda.sh"
if [ ! -f "${CONDA_SH}" ]; then
    echo "ERROR: conda not found at ${CONDA_SH}" >&2
    exit 1
fi
source "${CONDA_SH}"
conda activate /homes/observatorio/juagudeloo/.conda/envs/pytorch_jupyter
# The system libstdc++ is older than the one conda's PIL needs; without this the import fails.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Qwen3-8B was downloaded once into the pipeline repo's cache; stay offline so a missing file
# fails immediately instead of starting a 16 GB download on a compute node.
export HF_HOME="/scratchsan/observatorio/juagudeloo/Doctorado/sdo-paper-to-observation/models"
export HF_HUB_OFFLINE=1

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: config not found: ${CONFIG}" >&2
    exit 1
fi
if [ ! -f scripts/train_projector.py ]; then
    echo "ERROR: scripts/train_projector.py does not exist yet (docs/surya_forward_pass_study_plan.md, section 11)" >&2
    exit 1
fi

echo "Job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)}, config ${CONFIG}"
nvidia-smi -L 2>/dev/null | head -1
python3 -u scripts/train_projector.py --config "${CONFIG}"
STATUS=$?
echo "Exit code: ${STATUS}"
exit ${STATUS}
