#!/bin/sh
#SBATCH --job-name=retrieval-datacomp-jp
#SBATCH --output /home/skhanuja/vlr-project/data/scratch_retrieval-datacomp-jp
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time 2-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=skhanuja@andrew.cmu.edu

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate clip-retrieval-env

echo ${HOSTNAME}
export HF_HOME=/scratch/${USER}/cache

countries=("india")

for country in "${countries[@]}"
do
    python /home/skhanuja/vlr-project/data/retrieval_datacomp.py --config /home/skhanuja/vlr-project/configs/retrieval_datacomp.yaml
done
