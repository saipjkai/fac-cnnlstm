#- FOR REPRODUCIBLE RESULTS ##### 
print('Running in 1-thread CPU mode for fully reproducible results training a CNN and generating numpy randomness.  This mode may be slow...')

# Seed value
# Apparently you may use different seed values at each stage
seed_value=42

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

import pickle

from sklearn.model_selection import train_test_split

from process_data import DataGenerator
from C3D import c3d_model

from keras.callbacks import ModelCheckpoint, EarlyStopping 

from plot_utils import plot_history
#- LIBS & FRAMEWORKS #####


#- MAIN #####
def main(args, save_to_backup=True):
    # Save to Backup - weights, training curves
    if save_to_backup:
        version = args.version
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


    # train, validation & test datasets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=seed_value, stratify=y)

    # training & validation dataloaders
    batch_size=2
    train_loader = DataGenerator(X_train, y_train, batch_size=batch_size, dims=(16, 112, 112, 3), model_arch="c3d", num_classes=NUM_CLASSES)
    validation_loader = DataGenerator(X_valid, y_valid, batch_size=batch_size, dims=(16, 112, 112, 3), model_arch="c3d", num_classes=NUM_CLASSES)

    # training
    epochs = 20
    lr = 1e-4

    if save_to_backup:
        # Model - callbacks
        best_model_cb = ModelCheckpoint(weights_version_path, save_best_only=True, monitor='val_loss', verbose=1)
        callbacks = [best_model_cb]
    else:
        callbacks=[]

    model = c3d_model(input_shape=(16, 112, 112, 3), output_classes=NUM_CLASSES)
    model.summary()
    
    history = model.fit(train_loader,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=validation_loader,
                        verbose=1)
    
    plot_history(history, save_path=history_plot_version_path)
#- MAIN #####


#- ARGS #####
def get_args():
    ap = ArgumentParser()
    ap.add_argument('--pkl_dir', help="path to pickled dataset directory", required=True)
    ap.add_argument('--version', help="version number", required=True)
    args = ap.parse_args()
    return args
#- ARGS #####


if __name__ == "__main__":
    args = get_args()

    # base directory
    BASE_DIR = os.getcwd()

    # train data
    PKL_DIR = os.path.abspath(args.pkl_dir)
    X_train_pkl_path = os.path.join(PKL_DIR, 'X_train.pkl')
    with open(X_train_pkl_path, 'rb') as f:
        X = pickle.load(f)
    y_train_pkl_path = os.path.join(PKL_DIR, 'y_train.pkl')
    with open(y_train_pkl_path, 'rb') as f:
        y = pickle.load(f)

    # classes
    CLASSES_PATH = os.path.join(PKL_DIR, 'classes.pkl')
    with open(CLASSES_PATH, 'rb') as f:
        CLASSES = pickle.load(f)
    NUM_CLASSES = len(CLASSES)

    # backup
    BACKUP_DIR = os.path.join(BASE_DIR, "backup")
    if not os.path.isdir(BACKUP_DIR):
        os.mkdir(BACKUP_DIR)
    
    main(args)