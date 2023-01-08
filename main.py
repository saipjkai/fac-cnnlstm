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
from argparse import ArgumentParser
from datetime import date


from data_utils import get_Xy, DataGenerator 
from sklearn.model_selection import train_test_split

from CNN_LSTM import create_model
from keras.callbacks import ModelCheckpoint, EarlyStopping 
from plot_utils import plot_history
#- LIBS & FRAMEWORKS #####

import os
import pickle
import numpy as np

from keras.utils import Sequence
from keras.utils import to_categorical


#- TRAINING #####
def train(X, y, model, version, batch_size=1, no_epochs=10, verbose=1, save_to_backup=True):
    # Save to Backup - weights, training curves
    if save_to_backup:
        todays_date = date.today().isoformat()
        training_backup_path = os.path.join(BACKUP_DIR, todays_date)
        if not os.path.isdir(training_backup_path):
            os.mkdir(training_backup_path)

        weights_path = os.path.join(training_backup_path, "weights")
        if not os.path.isdir(weights_path):
            os.mkdir(weights_path)

        curves_path = os.path.join(training_backup_path, "curves")
        if not os.path.isdir(curves_path):
            os.mkdir(curves_path)

        weights_version = "weights_{}_v{}_tf.h5".format(todays_date, version)
        weights_version_path = os.path.join(weights_path, weights_version)

        history_plot_version = "accVSval_{}_v{}.png".format(todays_date, version)
        history_plot_version_path = os.path.join(curves_path, history_plot_version) 
    
    # Data - ready
    # split data 
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, shuffle=True, random_state = seed_value)
    # data generators
    train_gen = DataGenerator(X_train, y_train, batch_size=batch_size)
    valid_gen = DataGenerator(X_valid, y_valid, batch_size=batch_size)
    
    # Model - callbacks
    best_model_cb = ModelCheckpoint(weights_version_path, save_best_only=True, monitor='val_loss', verbose=verbose)
    # early_stopping_cb = EarlyStopping(monitor='val_loss', patience=3, verbose=verbose, restore_best_weights=True)
    callbacks = [best_model_cb]

    # Model - training
    history = model.fit(train_gen,
                        epochs=no_epochs,
                        verbose=verbose,
                        validation_data=valid_gen,
                        callbacks=callbacks
                    )
    
    # Model - plot history & save
    plot_history(history, save_path=history_plot_version_path)
#- TRAINING #####


#- ARGS #####
def get_args():
    # args
    ap = ArgumentParser()
    ap.add_argument("--v", help='weights version number', required=True)
    ap.add_argument("--dims", help="single unit data dimensions => (Depth, Height, Width, Channels)", default=(50, 128, 224, 3))
    args = ap.parse_args()
    return args
#- ARGS #####


#- MAIN #####
if __name__ == "__main__":
    # ARGS
    args = get_args()

    # PATHS
    # root directory
    BASE_DIR = os.path.abspath(".")
    # backup directory
    BACKUP_DIR = os.path.join(BASE_DIR, "backup")
    if not os.path.isdir(BACKUP_DIR):
            os.mkdir(BACKUP_DIR)

    # data directory
    DATA_PATH = os.path.join(BASE_DIR, "data", "pickle")

    # MODEL INPUTS & OUTPUTS, VERSION
    # input dims
    DIMS = args.dims
    # output classes 
    CLASSES = {'Corner':0, 'Throw_in':1, 'Yellow_card':2}
    NO_CLASSES = len(CLASSES)
    # version 
    version = args.v

    # DATA
    # load X & y 
    X, y = get_Xy(DATA_PATH, dims=DIMS, data_flag='train')
    
    # MODEL ARCHITECTURE
    model = create_model(DIMS, NO_CLASSES, learning_rate=1e-4)

    # Model - training
    train(X, y, model, version=version, batch_size=4, no_epochs=40, verbose=1, save_to_backup=True)
#- MAIN #####

