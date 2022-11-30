#- FOR REPRODUCIBLE RESULTS ##### 
print('Running in 1-thread CPU mode for fully reproducible results training a CNN and generating numpy randomness.  This mode may be slow...')

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
import keras
from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)
# for later versions:
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)
#- FOR REPRODUCIBLE RESULTS ##### 


#- LIBS & FRAMEWORKS #####
from tqdm import tqdm
import pickle

import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt

from keras import layers
from keras import Model
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, Dropout, LSTM, Input, TimeDistributed
from keras.applications.vgg16 import VGG16

from sklearn.model_selection import train_test_split

from argparse import ArgumentParser
from datetime import date

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
#- LIBS & FRAMEWORKS #####


#- MODEL CONFIGURATION #####
def create_model(dims, no_classes, learning_rate=1e-4):
    vggModel = VGG16(weights="imagenet", input_shape=dims[1:], include_top=False)
    for layer in vggModel.layers[:-1]:
        layer.trainable=False

    model = Sequential()
    input_layer = Input(shape=dims)
    model = TimeDistributed(vggModel)(input_layer) 
    model = TimeDistributed(Flatten())(model)
    
    model = LSTM(128, return_sequences=False)(model)
    model = Dropout(.5)(model)
    
    output_layer = Dense(no_classes, activation='softmax')(model)

    model = Model(input_layer, output_layer)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['accuracy'])

    return model
#- MODEL CONFIGURATION #####


#- PLOT MODEL #####
def plot_model_metrics(history):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.title('Model Performance for Video Action Detection', size=18, c="C7")
    plt.ylabel('Metrics', size=15, color='C7')
    plt.xlabel('Epoch No.', size=15, color='C7')

    plt.plot(history.history['accuracy'],  'o-', label='Training Data Accuracy', linewidth=2, c='C2') #C2 = green color.
    plt.plot(history.history['loss'],  '-', label='Training Loss', linewidth=2, c='C2') 
    plt.plot(history.history['val_accuracy'],  'o-', label='Validation Data Accuracy', linewidth=2, c='r') #r = red color.
    plt.plot(history.history['val_loss'],  '-', label='Validation Loss', linewidth=2, c='r') 
    
    plt.legend()    

    plt_path = os.path.join(base_directory, "models", "accVSval_{}_v{}.png".format(date.today().isoformat(), args.v))
    plt.savefig(plt_path)

    plt.show()
    plt.close()
#- PLOT MODEL #####


#- MAIN #####
if __name__ == "__main__":
    # args
    ap = ArgumentParser()
    ap.add_argument("--v", help='weights version number', required=True)
    args = ap.parse_args()

    # classes
    classes = {'Corner':0, 'Throw_in':1, 'Yellow_card':2, 'Other': 3}
    no_classes = len(classes)

    # root directory
    base_directory = os.path.abspath(".")

    # model input dimensions
    D = 50   # New Depth size => Number of frames.
    W = 128  # New Frame Width.
    H = 128  # New Frame Height.
    C = 3    # Number of channels.
    dims = (D, W, H, C) # Single Video shape.

    # data root 
    train_path = os.path.join(base_directory, "data", "pickle")
    
    # load X & y 
    data_X_path = os.path.join(train_path, 'X_{}x{}_{}.pkl'.format(D, W, no_classes))
    with open(data_X_path, 'rb') as X_pkl:
        X = pickle.load(X_pkl)
    data_y_path = os.path.join(train_path, 'y_{}x{}_{}.pkl'.format(D, W, no_classes))
    with open(data_y_path, 'rb') as y_pkl:
        y = pickle.load(y_pkl)

    # Convert target vectors to categorical targets
    y = to_categorical(y).astype(np.int32)

    # Data split 
    x_t, x_v, y_t, y_v = train_test_split(X, y, test_size=0.10, shuffle=True, random_state = 42)

    # Model - parameters
    batch_size = 1
    no_epochs = 20
    learning_rate = 0.0001
    verbosity = 2

    # Model - architecture
    model = create_model(dims, no_classes, learning_rate=learning_rate)
    # model.summary()
    # exit(0)

    # Model - monitoring
    # early_stopping_cb = EarlyStopping(
    #                             monitor='val_loss',
    #                             patience=3,
    #                             verbose=2,
    #                             restore_best_weights=True
    #        )
    weights_path = os.path.join(base_directory, "models", "weights_{}_v{}_tf.h5".format(date.today().isoformat(), args.v))
    best_model_cb = ModelCheckpoint(weights_path, save_best_only=True, monitor='val_loss', verbose=2)
    callbacks = [best_model_cb]

    # Model - training
    history = model.fit(x_t, y_t,
                        batch_size=batch_size,
                        epochs=no_epochs,
                        verbose=verbosity,
                        validation_data=(x_v, y_v),
                        callbacks=callbacks
                    )

    # Model - save weights
    # model.save(weights_path)

    # plot & save metrics 
    plot_model_metrics(history)
#- MAIN #####

