import pickle
import argparse
import logging
import json
import sys
import numpy as np
from data import load_data
from keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report


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

    parser = argparse.ArgumentParser(description="""
    Script that loads model and it's tokenizer,
    and evaluates it on test data.
    Computes precision recall and f1 both per class and macro average.
    Saves the computed metrics to the file and prints report to the stdout.
    """)
    parser.add_argument('model', help='classifier model')
    parser.add_argument('tokenizer', help='tokenizer used with model')
    parser.add_argument('test_data', help='data to test on')
    parser.add_argument('--output_file', help='file where to save the results')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load the model,
    # and the corresponding tokenizer.
    logging.info("loading model...")
    with open(args.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)

    model = load_model(args.model)
    logging.info("done")
    print(model.summary())

    # Load and preprocess dataset
    logging.info("loading and preprocessing dataset...")
    (x_test, y_test), (tokenizer, label_encoder) = load_data(args.test_data, tokenizer=tokenizer, maxlen=50)

    # Evaluate model on test set
    logging.info("evaluating model...")
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    logging.info("done")

    print(classification_report(y_true, y_pred))

    if args.output_file:
        metrics = get_metrics(y_pred, y_true)
        metrics['labels'] = label_encoder.classes_.tolist()
        with open(args.output_file, "w") as f:
            json.dump(metrics, f)
