import argparse
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--max_words', type=int, default=20000)
    parser.add_argument('--max_len', type=int, default=400)
    parser.add_argument('--gpu_count', type=int, default=os.environ['SM_NUM_GPUS'])

    # input data and model directories
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    max_words  = args.max_words
    max_len    = args.max_len
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.train
    testing_dir   = args.test
    
    training_data = pd.read_csv(training_dir+'/train.csv',sep=',',header=None)
    testing_data = pd.read_csv(testing_dir+'/test.csv',sep=',',header=None)
    
    data = pd.concat([training_data,testing_data], ignore_index=True)
    features = data[0].values
    labels = data[1].values
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(features)
    sequences = tokenizer.texts_to_sequences(features)
    features = pad_sequences(sequences, maxlen=max_len)

    vocab_size = len(tokenizer.word_index) 
    
    #Splitting the data again, because we needed to concat it before to train the tokenizer
    X_train, X_test, y_train, y_test = train_test_split(features,labels, random_state=0)

    # Building the model
    model = Sequential()
    model.add(layers.Embedding(max_words, 40))
    model.add(layers.LSTM(40,dropout=0.5))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('Model accuracy: ',test_acc)
    
    model_path = '{}/{}/00000001'.format(model_dir, 'depression_classifier')
    tf.saved_model.save(model, model_path)
