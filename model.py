from keras.layers import Conv1D, MaxPool1D, Concatenate, Dropout, Dense
from keras.layers import Input, GlobalMaxPool1D, Reshape, ReLU
from keras.layers import ELU, LeakyReLU, Activation
from keras.models import Model
from keras.optimizers import Adadelta


def build_baseline_model(input_size, embedding, windows=[3, 4, 5], filter_size=100,
                         dropout_rate=0.5, lr=1.0, activation='relu'):
    """Builds CNN model"""

    inputs = Input(shape=(input_size,))

    emb = embedding(inputs)

    regions = []
    for window in windows:
        conv = Conv1D(filter_size, window)(emb)

        if activation in ('relu', 'tanh', 'sigmoid'):
            conv = Activation(activation)(conv)
        elif activation in ('leakyrelu'):
            conv = LeakyReLU()(conv)
        elif activation in ('elu'):
            conv = ELU()(conv)
        else:
            raise ValueError('unknown activation')

        pool = GlobalMaxPool1D()(conv)

        regions.append(pool)

    conc = Concatenate(axis=1)(regions) if len(regions) > 1 else regions[0]
    drop = Dropout(dropout_rate)(conc)
    outputs = Dense(2, activation='softmax')(drop)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adadelta(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
