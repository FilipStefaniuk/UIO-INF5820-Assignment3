#!/bin/bash

#SBATCH --job-name=best-inf5820
#SBATCH --account=nn9447k
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10000
#SBATCH --output=./logs/best.log

module purge
module use -a /projects/nlpl/software/modulefiles/
module add nlpl-inf5820
module add nlpl-tensorflow

SEED=123
NUM_ITERS=10
WORD_VECTORS="/usit/abel/u1/filipste/vectors/GoogleNews-vectors-negative300.bin"
OUTPUT_DIR="./results/baseline/seed/"

mkdir -p $OUTPUT_DIR

python train_model.py --mode=multichannel --results_path="{$OUTPUT_DIR}best_glove.json" --word_vectors="/usit/abel/u1/filipste/vectors/glove.840B.300d.bin" --seed="$SEED" --model_tmp_path="best.model.tmp" --batch_size=512
