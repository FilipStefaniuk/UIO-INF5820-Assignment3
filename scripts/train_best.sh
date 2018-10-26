#!/bin/bash

#SBATCH --job-name=best-inf5820
#SBATCH --account=nn9447k
#SBATCH --time=05:00:00
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=30000
#SBATCH --output=./logs/best.log

module purge
module use -a /projects/nlpl/software/modulefiles/
module add nlpl-inf5820
module add nlpl-tensorflow

SEED=123
WORD_VECTORS="/usit/abel/u1/filipste/vectors/wiki.en.fasttext.bin"
OUTPUT_DIR="./results/best/"

mkdir -p $OUTPUT_DIR

for ((i=0; i < 3; i++)); do
	OUTPUT_FILE="${OUTPUT_DIR}best_${i}.json"
	python train_model.py --mode=multichannel --results_path="$OUTPUT_FILE" --word_vectors="$WORD_VECTORS" --seed="$SEED"  --model_tmp_path="best.model.tmp" --save_path="./best_${i}.model" --epochs=50 --patience=10
done
