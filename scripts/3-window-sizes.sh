#!/bin/bash

#SBATCH --job-name=window-inf5820
#SBATCH --account=nn9447k
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10000
#SBATCH --output=./logs/windows.log

module purge
module use -a /projects/nlpl/software/modulefiles/
module add nlpl-inf5820
module add nlpl-tensorflow

SEED=123
WORD_VECTORS="/usit/abel/u1/filipste/vectors/GoogleNews-vectors-negative300.bin"
OUTPUT_DIR="./results/baseline/windows/"
WINDOW_SIZES=(
    "1"
    "3"
    "5"
    "2 3 4"
    "3 4 5"
    "4 5 6"
    "7 8 9"
    "14 15 16"
    "2 3 4 5"
    "6 7 8 9"
    "1 3 5 7"
)

mkdir -p $OUTPUT_DIR

for ((i=0; i < ${#WINDOW_SIZES[@]}; i++)); do
    OUTPUT_FILE="${OUTPUT_DIR}$(echo ${WINDOW_SIZES[$i]} | sed 's/ /-/g').json"
    python train_model.py --mode=static --results_path="$OUTPUT_FILE" --seed="$SEED" --model_tmp_path="windows.model.tmp" --windows ${WINDOW_SIZES[$i]}
done
