from keras.layers import Conv1D, MaxPool1D, Concatenate, Dropout, Dense
from keras.layers import Input, GlobalMaxPool1D, Reshape, ReLU, BatchNormalization
from keras.layers import ELU, LeakyReLU, Activation, Add
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


def build_deep_model(input_size, embedding):

        inputs = Input(shape=(input_size,))
        emb = embedding(inputs)

        conv1_1 = Conv1D(200, 2, padding='same')(emb)
        conv1_1 = BatchNormalization()(conv1_1)
        conv1_1 = LeakyReLU()(conv1_1)

        conv1_2 = Conv1D(200, 3, padding='same')(emb)
        conv1_2 = BatchNormalization()(conv1_2)
        conv1_2 = LeakyReLU()(conv1_2)

        conv1_3 = Conv1D(200, 4, padding='same')(emb)
        conv1_3 = BatchNormalization()(conv1_3)
        conv1_3 = LeakyReLU()(conv1_3)

        conc1 = Concatenate()([conv1_1, conv1_2, conv1_3])

        conv2_1 = Conv1D(200, 2, padding='same')(conc1)
        conv2_1 = BatchNormalization()(conv2_1)
        conv2_1 = LeakyReLU()(conv2_1)

        conv2_2 = Conv1D(200, 3, padding='same')(conc1)
        conv2_2 = BatchNormalization()(conv2_2)
        conv2_2 = LeakyReLU()(conv2_2)

        conv2_3 = Conv1D(200, 4, padding='same')(conc1)
        conv2_3 = BatchNormalization()(conv2_3)
        conv2_3 = LeakyReLU()(conv2_3)

        conc2 = Concatenate()([conv2_1, conv2_2, conv2_3])
        add2 = Add()([conc1, conc2])

        conv3_1 = Conv1D(200, 2, padding='same')(add2)
        conv3_1 = BatchNormalization()(conv3_1)
        conv3_1 = LeakyReLU()(conv3_1)

        conv3_2 = Conv1D(200, 3, padding='same')(add2)
        conv3_2 = BatchNormalization()(conv3_2)
        conv3_2 = LeakyReLU()(conv3_2)

        conv3_3 = Conv1D(200, 4, padding='same')(add2)
        conv3_3 = BatchNormalization()(conv3_3)
        conv3_3 = LeakyReLU()(conv3_3)

        conc3 = Concatenate()([conv3_1, conv3_2, conv3_3])
        add3 = Add()([add2, conc3])

        pool = GlobalMaxPool1D()(add3)
        drop = Dropout(0.6)(pool)
        outputs = Dense(2, activation='softmax')(drop)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        return model
