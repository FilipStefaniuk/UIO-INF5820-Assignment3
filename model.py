from keras.layers import Conv2D, Conv1D, MaxPool1D, Concatenate, Dropout, Dense, Input, GlobalMaxPool1D, Reshape
from keras.models import Model
from keras.optimizers import Adadelta


def build_model(input_size, embedding, windows=[3, 4, 5], filter_size=100, dropout_rate=0.5, lr=1.0):

    inputs = Input(shape=(input_size,))

    emb = embedding(inputs)

    regions = []
    for window in windows:
        conv = Conv2D(filter_size, (window, 300), activation='relu')(emb)
        reshape = Reshape((-1, filter_size))(conv)
        pool = GlobalMaxPool1D()(reshape)

        regions.append(pool)

    conc = Concatenate(axis=1)(regions) if len(regions) > 1 else regions[0]
    drop = Dropout(dropout_rate)(conc)
    outputs = Dense(2, activation='softmax')(drop)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adadelta(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
