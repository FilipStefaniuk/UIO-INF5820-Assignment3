#!/bin/bash

#SBATCH --job-name=infer-oov-inf5820
#SBATCH --account=nn9447k
#SBATCH --time=05:00:00
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=30000
#SBATCH --output=./logs/infer-oov.log

module purge
module use -a /projects/nlpl/software/modulefiles/
module add nlpl-inf5820
module add nlpl-tensorflow


WORD_VECTORS="/usit/abel/u1/filipste/vectors/wiki.en.fasttext.bin"
OUTPUT_DIR="./results/baseline/infer-oov/"

mkdir -p $OUTPUT_DIR

OUTPUT_FILE="${OUTPUT_DIR}fasttext-infer-oov.json"
python train_model.py --mode=static --results_path="$OUTPUT_FILE" --word_vectors="$WORD_VECTORS"

