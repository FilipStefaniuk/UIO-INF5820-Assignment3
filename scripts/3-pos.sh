#!/bin/bash

#SBATCH --job-name=inf5820_assignment3
#SBATCH --account=nn9447k
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10000
#SBATCH --output=./results/baseline/pos/output.log
#SBATCH --err=./results/baseline/pos/err.log

module purge
module use -a /projects/nlpl/software/modulefiles/
module add nlpl-inf5820
module add nlpl-tensorflow

SEED=123
OUTPUT_DIR="./results/baseline/pos/"
POS_WORD_VECTORS=(
    "/projects/nlpl/data/vectors/11/1.zip"
    "/projects/nlpl/data/vectors/11/18.zip"
    "/projects/nlpl/data/vectors/11/20.zip"
)

mkdir -p $OUTPUT_DIR

for ((i=0; i < ${#POS_WORD_VECTORS[@]}; i++)); do
    OUTPUT_FILE="${OUTPUT_DIR}$(basename ${POS_WORD_VECTORS[i]}).json"
    python train_model.py --mode=static --results_path="$OUTPUT_FILE" --word_vectors="${POS_WORD_VECTORS[$i]}" --seed="$SEED" --pos
done