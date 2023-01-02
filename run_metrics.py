from argparse import ArgumentParser
import os
import numpy as np

from model_utils import load_model_from_weights, run_inference

from data_utils import load_pkl, get_Xy
from plot_utils import plot_cn_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt


def calculate_metrics(y_actual, y_prediction_probabilities, y_prediction):
    # AUC score
    auc = roc_auc_score(y_actual, y_prediction_probabilities, multi_class='ovr')
    print("AUC score: {}".format(auc))

    # Confusion matrix
    cn_matrix = confusion_matrix(y_actual, y_prediction, labels=[0, 1, 2])
    print("Confusion matrix: \n{}".format(cn_matrix))
    plot_cn_matrix(confusion_matrix=cn_matrix, labels=LABELS, save_path='cnf_matrix_{}.png'.format(WEIGHTS))
    
    # Accuracy
    accuracy = accuracy_score(y_actual, y_prediction)
    print("Accuracy: {}".format(accuracy))


def get_args():
    ap = ArgumentParser()
    ap.add_argument("--weights", help="model to use for calculating metrics", required=True)
    ap.add_argument("--dims", help="single unit data dimensions => (Depth, Height, Width, Channels)", default=(50, 128, 224, 3))
    args = ap.parse_args()
    return args


def main(args):
    #- Data - single unit dimensions = (D, H, W, C)
    DIMS = args.dims # Single Video shape.
    D = DIMS[0]  # Depth size => Number of frames.
    H = DIMS[1]  # Frame Height.
    W = DIMS[2]  # Frame Width.
    C = DIMS[3]  # Number of channels.

    #- Data - load test data
    X_test, y_test = getXy(TEST_DATA_PATH, DIMS, data_flag='test')

    #- Model - load weights
    model = load_model_from_weights(WEIGHTS_PATH)
    
    #- Model - inference
    y_actual, y_prediction_probabilities, y_prediction = run_inference(model, X_test, y_test, DIMS)

    #- Model - metrics
    calculate_metrics(y_actual, y_prediction_probabilities, y_prediction)


if __name__ == "__main__":
    args = get_args()

    #- Data - classes
    LABELS = ['Corner', 'Throw_in', 'Yellow_card']
    NO_CLASSES = len(LABELS)

    #- Data - paths 
    # base dir
    BASE_DIR = os.path.abspath(os.getcwd())
    # test data dir
    TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "pickle")

    #- Model - paths
    WEIGHTS = (args.weights).split('/')[-1]
    WEIGHTS_PATH = os.path.join(BASE_DIR, args.weights)

    main(args)



    



