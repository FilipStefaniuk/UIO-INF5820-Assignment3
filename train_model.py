import os
import sys
import json
import random
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from data import load_data
from emb import load_emb_model, get_emb_layer
from model import build_baseline_model

from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Concatenate, Embedding, Reshape, Input
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="""
        Train CNN classifier for sentiment classification task.
        Atchitecture is based on "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional
        Neural Networks for Sentence Classification" paper.
    """)

    parser.add_argument('--epochs', type=int, default=30, help="number of epochs to train.")
    parser.add_argument('--batch_size', type=int, default=256, help="training batch size.")
    parser.add_argument('--lr', type=float, default=1.0, help="learning rate used in adadelta optimizer.")
    parser.add_argument('--seed', type=int, help="random seed")
    parser.add_argument('--max_words', type=int, help="max number of words in dictionary")
    parser.add_argument('--max_len', type=int, default=50, help="sentences are padded/clipped to that length")
    parser.add_argument('--filter_size', type=int, default=100, help="number of filters in convolutions.")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="dropout rate.")
    parser.add_argument('--windows', nargs='+', type=int, default=[3, 4, 5], help="window sizes.")
    parser.add_argument('--patience', type=int, default=5, help="patience for early stopping.")

    parser.add_argument('--emb_dim', type=int, default=300,
                        help="dimentionality of word embeddings (overrided when loaded from file)")

    parser.add_argument('--mode', choices=['static', 'non-static', 'multichannel'], default='static',
                        help="mode of embedding layer.")

    parser.add_argument('--model_tmp_path', default='model.tmp.pkl',
                        help="tmp file where to save best model during trainig.")

    parser.add_argument('--words', choices=['raw', 'lemmatized', 'pos_tagged'], default='raw',
                        help="type of  words from dataset.")

    parser.add_argument('--activation', choices=['relu', 'tanh', 'sigmoid', 'leakyrelu', 'elu'], default='relu',
                        help="activation function.")

    parser.add_argument('--word_vectors', help="pretrained word embeddings to use.")
    parser.add_argument('--results_path', help="path where to save the json file with result metrics.")
    parser.add_argument('--save_path', help="path where to save model.")
    parser.add_argument('--graph_image_path', help="path where to save image with model architecture.")

    return parser.parse_args()


def get_embedding(mode, path, tokenizer, input_len, emb_dim=300):
    """Creates embedding layer accordingly to model's mode."""

    emb_model = load_emb_model(path) if path else None
    emb_dim = emb_model.wv.vector_size if emb_model else emb_dim

    if mode in ('static'):
        return get_emb_layer(emb_model, tokenizer, trainable=False)

    elif mode in ('non-static'):
        return get_emb_layer(emb_model, tokenizer, trainable=True)

    elif mode in ('multichannel'):

        inputs = Input(shape=(input_len,))

        emb1 = get_emb_layer(emb_model, tokenizer, trainable=False)(inputs)
        emb2 = get_emb_layer(emb_model, tokenizer, trainable=True)(inputs)

        outputs = Concatenate()([emb1, emb2])

        emb = Model(inputs=inputs, outputs=outputs)
        emb.name = 'embedding_1'
        return emb

    raise ValueError('invalid mode')


def get_metrics(y_true, y_pred):
    """Computes metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    avg_prec, avg_rec, avg_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    return {
        'acc': acc,
        'prec': prec.tolist(),
        'rec': rec.tolist(),
        'f1': f1.tolist(),
        'avg_prec': avg_prec,
        'avg_rec': avg_rec,
        'avg_f1': avg_f1
    }


if __name__ == '__main__':

    # Create logger
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    args = get_args()
    logger.info("training model in {} mode".format(args.mode))

    # if seed is provided in arguments,
    # set random state
    if args.seed:
        logger.info("setting random seed to {}".format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    # Load data from files (training and validation), tokenize and preprocess texts
    logger.info("loading and preprocessing dataset")
    train, (tokenizer, label_encoder) = load_data('./data/stanford_sentiment_binary_train.tsv.gz',
                                                  maxlen=args.max_len, max_words=args.max_words, words=args.words)

    val, _ = load_data('./data/stanford_sentiment_binary_dev.tsv.gz', tokenizer=tokenizer,
                       label_encoder=label_encoder, maxlen=args.max_len, max_words=args.max_words, words=args.words)

    logger.info("training: {} samples, validation: {} samples".format(len(train[0]), len(val[0])))
    logger.info("using {} {} words".format(tokenizer.num_words or len(tokenizer.word_index), args.words))

    # Create embedding layer
    logger.info("building embedding layer")
    emb = get_embedding(args.mode, args.word_vectors, tokenizer, args.max_len, emb_dim=args.emb_dim)

    # Build model
    logger.info("building model")
    model = build_baseline_model(args.max_len, emb, windows=args.windows, filter_size=args.filter_size, lr=args.lr,
                                 dropout_rate=args.dropout_rate, activation=args.activation)

    model.summary()

    callbacks = [
        EarlyStopping(patience=args.patience),
        ReduceLROnPlateau(patience=2),
        ModelCheckpoint(args.model_tmp_path, save_best_only=True)
    ]

    # Train model
    try:
        logger.info("training the model")
        model.fit(train[0], train[1], epochs=args.epochs, validation_data=val,
                  batch_size=args.batch_size, callbacks=callbacks)
    except KeyboardInterrupt:
        logger.error("keyboard interrupt")
        if os.path.isfile(args.model_tmp_path):
            os.remove(args.model_tmp_path)
        sys.exit()

    # Load best model and remove tmp file
    if os.path.isfile(args.model_tmp_path):
        logger.info("loading best model weights")
        model.load_weights(args.model_tmp_path)
        os.remove(args.model_tmp_path)

    y_val_preds = model.predict(val[0])
    print(classification_report(np.argmax(val[1], axis=1),
          np.argmax(y_val_preds, axis=1), target_names=label_encoder.classes_))

    # Save metrics
    if args.results_path:
        logger.info("computing and saving metrics")
        metrics = get_metrics(np.argmax(val[1], axis=1), np.argmax(y_val_preds, axis=1))
        metrics['labels'] = label_encoder.classes_.tolist()

        with open(args.results_path, "w") as f:
            json.dump(metrics, f)

    # Plot model architecture to a file.
    if args.graph_image_path:
        logger.info("saving model architecure plot")
        plot_model(model, args.graph_image_path, show_shapes=True)

    # Save model
    if args.save_path:
        logger.info("saving model")
        model.save(args.save_path)
        with open('./tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
