import os
import json
import argparse

import numpy as np

from model import build_model, save_weights

data_dir = '/content/drive/My Drive/Music Generation/data'
log_dir = '/content/drive/My Drive/Music Generation/logs'

batch_size = 16  #size of batch
seq_length = 64  #length of char sequence

class TrainLogger():
  def __init__(self, file):
    self.file = os.path.join(log_dir, file) #creating a log in dir, given the  filename
    self.epochs = 0
    with open(self.file, 'w') as fp:
      fp.write('epoch, loss, acc\n')

  def add_entry(self, loss, acc):
    self.epochs += 1
    s = '{}, {}, {}\n'.format(self.epochs, loss, acc)
    with open(self.file, 'a') as fp:
      fp.write(s)

def read_batches(data, vocab_size):
  length = data.shape[0];
  total_batchs = int(length / batch_size);

  for batch_index in range(0, total_batchs - seq_length, seq_length):
    X = np.zeros((batch_size, seq_length))
    Y = np.zeros((batch_size, seq_length, vocab_size))

    for row_index in range(0, batch_size):
      for char in range(0, seq_length):
        X[row_index, char] = data[total_batchs * row_index + batch_index + char]
        Y[row_index, char, data[total_batchs * row_index + batch_index + char + 1]] = 1

      yield X, Y
     
def train(text, epochs=100, save_freq=10):
  char_to_idx = { ch: i for (i, ch) in enumerate(sorted(list(set(text))))}

  print("Number of unique chars: " + str(len(set(char_to_idx))))
   
  #saving that dict as json file
  with open(os.path.join(data_dir, 'char_to_idx.json'), 'w') as fp:
    json.dump(char_to_idx, fp)

  #inv of char_to_idx
  idx_of_char = { i: ch for (ch, i) in char_to_idx.items()}
  vocab_size = len(char_to_idx)

  #model_architecture
  model = build_model(batch_size, seq_length, vocab_size)
  model.summary()
  model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
  T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32)

  print("Length of text: " + str(T.size))

  steps_per_epoch = (len(text) / batch_size - 1 ) / seq_length

  log = TrainLogger('training_log.csv')

  for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))

    losses, accs = [], []


    for i, (X,Y) in enumerate(read_batches(T, vocab_size)):
      print(X);
      
      # training the model built above with the respective individual batches
      loss, acc = model.train_on_batch(X, Y)
      print('Batch {}: loss = {}, acc = {}'.format(i + 1, loss, acc))
      losses.append(loss)
      accs.append(acc)
      
    # adding the average of loss,acc for that respective epoch all batches training to log
    log.add_entry(np.average(losses), np.average(accs))
    
    # if epoch reach multiple of 10 then save the model
    if (epoch +1) % save_freq == 0:
      save_weights(epoch + 1, model)
      print('Saved checkpoint to ', 'weights.{}.h5'.format(epoch + 1))
      
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Train the model on some text.')
  parser.add_argument('--input', default = 'data/gigs_tune.txt' , help= 'name of the text file to train from')
  parser.add_argument('--epochs', type = int, default = 100, help = 'number of epochs to  train for')
  parser.add_argument('--freq', type = int, default = 10 , help = 'checkpoint save frequency')

  args = parser.parse_args()

  # create the folder for log file
  if not os.path.exists(log_dir):
     os.makedirs(log_dir)
  
  # train(open('gigs_tune.txt').read(), 100, 10)
  train(open(args.input).read(), args.epochs, args.freq)
  
