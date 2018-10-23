from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd


def load_data(path, tokenizer=None, label_encoder=None, pos=False, maxlen=None, max_words=None):
    """Loads SST dataset."""

    col_x = 'lemmatized' if pos else 'tokens'
    col_y = 'label'

    data = pd.read_csv(path, sep='\t', compression='gzip', usecols=[col_x, col_y])

    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(data[col_x])

    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(data[col_y])

    x_data = tokenizer.texts_to_sequences(data[col_x])
    x_data = pad_sequences(x_data, maxlen=maxlen, truncating='post', padding='post')

    y_data = to_categorical(label_encoder.transform(data[col_y]))

    return (x_data, y_data), (tokenizer, label_encoder)
