from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd


def load_data(path, tokenizer=None, label_encoder=None, words='raw', maxlen=None, max_words=None):
    """Loads SST dataset."""

    col_x = 'tokens' if words in ('raw') else 'lemmatized'
    col_y = 'label'

    data = pd.read_csv(path, sep='\t', compression='gzip', usecols=[col_x, col_y])

    if words in ('lemmatized'):
        data[col_x] = df[col_x].map(lambda x: " ".join(word.split('_')[0] for word in x.split()))

    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_words, filters='', lower=False)\
                    if words in ('pos_tagged') else Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(data[col_x])

    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(data[col_y])

    x_data = tokenizer.texts_to_sequences(data[col_x])
    x_data = pad_sequences(x_data, maxlen=maxlen, truncating='post', padding='post')

    y_data = to_categorical(label_encoder.transform(data[col_y]))

    return (x_data, y_data), (tokenizer, label_encoder)
