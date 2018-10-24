#!/bin/bash

#SBATCH --job-name=activation-inf5820
#SBATCH --account=nn9447k
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10000
#SBATCH --output=./logs/filters.log

module purge
module use -a /projects/nlpl/software/modulefiles/
module add nlpl-inf5820
module add nlpl-tensorflow

SEED=123
WORD_VECTORS="/usit/abel/u1/filipste/vectors/GoogleNews-vectors-negative300.bin"
OUTPUT_DIR="./results/baseline/activation/"
ACTIVATIONS=("relu", "tanh", "sigmoid", "leakyrelu", "elu")

mkdir -p $OUTPUT_DIR

for ((i=0; i < ${#ACTIVATIONS[@]}; i++)); do
    OUTPUT_FILE="${OUTPUT_DIR}${ACTIVATIONS[$i]}.json"
    python train_model.py --mode=static --results_path="$OUTPUT_FILE" --word_vectors="$WORD_VECTORS" --seed="$SEED" --activation="${ACTIVATIONS[$i]}" --model_tmp_path="activations.model.tmp"
done
