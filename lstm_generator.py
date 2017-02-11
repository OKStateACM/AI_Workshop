import os
import pickle
from six.moves import urllib

import tflearn
from tflearn.data_utils import *

path = "corpus.txt"
maxlen = 100

X, Y, char_idx = \
    textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=2)


input = tflearn.input_data([None, maxlen, len(char_idx)])   # input is a string of *maxlen* characters
lstm1 = tflearn.lstm(input, 256, return_seq=True)           # LSTM layer
dropout1 = tflearn.dropout(lstm1, 0.5)                      # dropout to avoid overfitting
lstm2 = tflearn.lstm(dropout1, 256)                         # LSTM layer
dropout2 = tflearn.dropout(lstm2, 0.5)                      # droupout to avoid overfitting
output = tflearn.fully_connected(dropout2, len(char_idx), activation='softmax')
optimizer = tflearn.regression(output, optimizer='adam', loss='categorical_crossentropy',
            learning_rate=0.001)

# Use TFlearn's sequence generator
model = tflearn.SequenceGenerator(optimizer, dictionary=char_idx,
        seq_maxlen=maxlen,
        clip_gradients=5.0,
        checkpoint_path='guten')

# load pretrained model
model.load('guten')

# and train!
for i in range(50):
    seed = random_sequence_from_textfile(path, maxlen)
    print("-- TESTING...")
    print("-- Test with temperature of 1.0 --")
    print(model.generate(600, temperature=1.0, seq_seed=seed))
    print("-- Test with temperature of 0.5 --")
    print(model.generate(600, temperature=0.5, seq_seed=seed))
    print("-- Test with temperature of 0.1 --")
    print(model.generate(600, temperature=0.1, seq_seed=seed))
    model.fit(X, Y, validation_set=0.2, batch_size=128,
            n_epoch=1, run_id='guten')
