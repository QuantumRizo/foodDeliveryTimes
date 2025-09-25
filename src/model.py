import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

def build_model(cat_dim, num_dim):
    cat_input = Input(shape=(cat_dim,), name='cat_input')
    cat_dense = Dense(32, activation='relu')(cat_input)
    num_input = Input(shape=(num_dim,), name='num_input')
    num_dense = Dense(16, activation='relu')(num_input)
    concat = Concatenate()([cat_dense, num_dense])
    x = Dense(64, activation='relu')(concat)
    x = Dense(32, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs=[cat_input, num_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model