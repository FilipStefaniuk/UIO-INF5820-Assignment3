#!/bin/bash

#SBATCH --job-name=multichannel-inf5820
#SBATCH --account=nn9447k
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10000
#SBATCH --output=./logs/multichannel.log

module purge
module use -a /projects/nlpl/software/modulefiles/
module add nlpl-inf5820
module add nlpl-tensorflow

SEED=123
OUTPUT_DIR="./results/baseline/multichannel/"
WORD_VECTORS=(
    "/usit/abel/u1/filipste/vectors/GoogleNews-vectors-negative300.bin"
    "/usit/abel/u1/filipste/vectors/glove.840B.300d.bin"
    "/usit/abel/u1/filipste/vectors/wiki.en.bin"
    "/projects/nlpl/data/vectors/11/17.zip"
    "/projects/nlpl/data/vectors/11/19.zip"
)

mkdir -p $OUTPUT_DIR

for ((i=0; i < ${#WORD_VECTORS[@]}; i++)); do
    OUTPUT_FILE="${OUTPUT_DIR}$(basename ${WORD_VECTORS[i]}).json"
    python train_model.py --mode=multichannel --results_path="$OUTPUT_FILE" --word_vectors="${WORD_VECTORS[$i]}" --seed="$SEED" --model_tmp_path="multichannel.model.tmp"
done
