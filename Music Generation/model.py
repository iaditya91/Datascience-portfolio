import os

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding

model_dir = 'model/'

def build_model(batch_size, seq_len, vocab_size):
  model = Sequential()
  model.add(Embedding(vocab_size, 512, batch_input_shape = (batch_size, seq_len)))

  for i in range(3):
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))

  # last but one layer containes vocab_size neurons
  model.add(TimeDistributed(Dense(vocab_size)))
  model.add(Activation('softmax'))
  return model
  
def load_weights(epoch, model):
  model.load_weights(os.path.join(model_dir, 'weights.{}.h5'.format(epoch)))

def save_weights(epoch, model):
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  model.save_weights(os.path.join(model_dir, 'weights.{}.h5'.format(epoch)))
