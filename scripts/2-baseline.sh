#!/bin/bash

#SBATCH --job-name=baseline-inf5820
#SBATCH --account=nn9447k
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10000
#SBATCH --output=./logs/baseline.log

module purge
module use -a /projects/nlpl/software/modulefiles/
module add nlpl-inf5820
module add nlpl-tensorflow

NUM_ITERS=10
WORD_VECTORS="/usit/abel/u1/filipste/vectors/GoogleNews-vectors-negative300.bin"
OUTPUT_DIR="./results/baseline/noseed/"

mkdir -p $OUTPUT_DIR

for ((i=1; i <= NUM_ITERS; i++)); do
    OUTPUT_FILE="${OUTPUT_DIR}results_${i}.json"
    python train_model.py --mode=static --results_path="$OUTPUT_FILE" --word_vectors="$WORD_VECTORS" --model_tmp_path="baseline.model.tmp"
done
