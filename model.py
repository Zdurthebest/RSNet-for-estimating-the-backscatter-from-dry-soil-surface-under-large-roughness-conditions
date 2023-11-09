

'''
   RSNet   2023.2.26
'''

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Activation, Concatenate

def upperlstm(input):
    out = LSTM(16, return_sequences=True)(input)
    out = Dropout(0.2)(out)
    out = LSTM(32, return_sequences=True)(out)
    out = Dropout(0.2)(out)
    out = LSTM(64, return_sequences=True)(out)
    out = Dropout(0.2)(out)
    out = LSTM(128, return_sequences=False)(out)
    model = Model(inputs=input, outputs=out, name='upperlstm')
    # model.summary()
    return model

def lowerlstm(input):
    out = LSTM(16, return_sequences=True)(input)
    out = Dropout(0.2)(out)
    out = LSTM(32, return_sequences=True)(out)
    out = Dropout(0.2)(out)
    out = LSTM(64, return_sequences=True)(out)
    out = Dropout(0.2)(out)
    out = LSTM(128, return_sequences=False)(out)
    model = Model(inputs=input, outputs=out, name='lowerlstm')
    # model.summary()
    return model

def doublelstm():
    input_up = Input(shape=(2, 1))
    input_down = Input(shape=(4, 1))

    model_up = upperlstm(input_up)
    model_down = lowerlstm(input_down)

    out_up = model_up.output
    out_down = model_down.output

    outputs = Concatenate(axis=1)([out_up, out_down])

    outputs = Dense(32)(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation('sigmoid')(outputs)

    model = Model(inputs=[input_up, input_down], outputs=outputs, name='doublelstm')
    model.summary()
    return model


if __name__ == '__main__':
    doublelstm()
















