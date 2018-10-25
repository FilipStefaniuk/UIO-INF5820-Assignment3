# INF5820-Assignment3
Sentiment Analysis with Convolutional Neural Networks

## Overview

### Train model
```
usage: train_model.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                      [--lr LR] [--seed SEED] [--max_words MAX_WORDS]
                      [--max_len MAX_LEN] [--filter_size FILTER_SIZE]
                      [--dropout_rate DROPOUT_RATE]
                      [--windows WINDOWS [WINDOWS ...]] [--patience PATIENCE]
                      [--emb_dim EMB_DIM]
                      [--mode {static,non-static,multichannel}]
                      [--model_tmp_path MODEL_TMP_PATH]
                      [--words {raw,lemmatized,pos_tagged}]
                      [--activation {relu,tanh,sigmoid,leakyrelu,elu}]
                      [--word_vectors WORD_VECTORS]
                      [--results_path RESULTS_PATH] [--save_path SAVE_PATH]
                      [--graph_image_path GRAPH_IMAGE_PATH]

Train CNN classifier for sentiment classification task. Atchitecture is based
on "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional
Neural Networks for Sentence Classification" paper.

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs to train.
  --batch_size BATCH_SIZE
                        training batch size.
  --lr LR               learning rate used in adadelta optimizer.
  --seed SEED           random seed
  --max_words MAX_WORDS
                        max number of words in dictionary
  --max_len MAX_LEN     sentences are padded/clipped to that length
  --filter_size FILTER_SIZE
                        number of filters in convolutions.
  --dropout_rate DROPOUT_RATE
                        dropout rate.
  --windows WINDOWS [WINDOWS ...]
                        window sizes.
  --patience PATIENCE   patience for early stopping.
  --emb_dim EMB_DIM     dimentionality of word embeddings (overrided when
                        loaded from file)
  --mode {static,non-static,multichannel}
                        mode of embedding layer.
  --model_tmp_path MODEL_TMP_PATH
                        tmp file where to save best model during trainig.
  --words {raw,lemmatized,pos_tagged}
                        type of words from dataset.
  --activation {relu,tanh,sigmoid,leakyrelu,elu}
                        activation function.
  --word_vectors WORD_VECTORS
                        pretrained word embeddings to use.
  --results_path RESULTS_PATH
                        path where to save the json file with result metrics.
  --save_path SAVE_PATH
                        path where to save model.
  --graph_image_path GRAPH_IMAGE_PATH
                        path where to save image with model architecture.
```

### Eval on test
```
usage: eval_on_test.py [-h] [--output_file OUTPUT_FILE]
                       model tokenizer test_data

Script that loads model and it's tokenizer, and evaluates it on test data.
Computes precision recall and f1 both per class and macro average. Saves the
computed metrics to the file and prints report to the stdout.

positional arguments:
  model                 classifier model
  tokenizer             tokenizer used with model
  test_data             data to test on

optional arguments:
  -h, --help            show this help message and exit
  --output_file OUTPUT_FILE
                        file where to save the results
```