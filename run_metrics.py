from argparse import ArgumentParser
import os
import numpy as np

from model_utils import load_model_from_weights

from process_data import batch_generator, DataGenerator
from plot_utils import plot_cn_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import pickle


def calculate_metrics(y_actual, y_prediction_probabilities, y_prediction):
    # AUC score
    auc = roc_auc_score(y_actual, y_prediction_probabilities, multi_class='ovo')
    print("AUC score: {}".format(auc))

    # Confusion matrix
    cn_matrix = confusion_matrix(y_actual, y_prediction, labels=[0, 1, 2])
    print("Confusion matrix: \n{}".format(cn_matrix))
    
    plot_cn_matrix(cnfMatrix=cn_matrix, labels=classes, save_path=CNF_MATRIX_PATH)
    
    # Accuracy
    accuracy = accuracy_score(y_actual, y_prediction)
    print("Accuracy: {}".format(accuracy))


def run_inference(model, X, y):
    test_dataloader = DataGenerator(X, y, batch_size=1, dims=(50, 128, 224, 3), num_classes=no_classes)

    y_actual= []
    y_prediction_probabilities = []
    y_prediction = []
    for x_t, y_t in test_dataloader:
        # predictions
        prediction = model.predict(x_t)
        y_prediction_probabilities.append(prediction[0].tolist())
        y_prediction.append(np.argmax(prediction[0]))

        # groundtruths
        y_actual.append(np.argmax(y_t[0]))

    return y_actual, y_prediction_probabilities, y_prediction


def get_args():
    ap = ArgumentParser()
    ap.add_argument("--weights", help="model to use for calculating metrics", required=True)
    # ap.add_argument("--dims", help="single unit data dimensions => (Depth, Height, Width, Channels)", default=(50, 128, 224, 3))
    ap.add_argument("--test_path", help="path to test data", required=True)
    args = ap.parse_args()
    return args


def main(args):
    #- Model - load weights
    model = load_model_from_weights(WEIGHTS_PATH)
    
    #- Model - inference
    y_actual, y_prediction_probabilities, y_prediction = run_inference(model, X_test, y_test)

    #- Model - metrics
    calculate_metrics(y_actual, y_prediction_probabilities, y_prediction)


if __name__ == "__main__":
    args = get_args()

    #- Data - paths 
    # base dir
    BASE_DIR = os.path.abspath(os.getcwd())
    # test data 
    TEST_DATA_PATH = os.path.join(BASE_DIR, args.test_path)

    #- Model - paths
    WEIGHTS_PATH = os.path.join(BASE_DIR, args.weights)
    WEIGHTS = (args.weights).split('/')[-1]
    
     #- Data - load test data and classes
    X_test_pkl_path = os.path.join(TEST_DATA_PATH, 'X_test.pkl')
    with open(X_test_pkl_path, 'rb') as f:
        X_test = pickle.load(f)
    y_test_pkl_path = os.path.join(TEST_DATA_PATH, 'y_test.pkl')
    with open(y_test_pkl_path, 'rb') as f:
        y_test = pickle.load(f)
    classes_path = os.path.join(TEST_DATA_PATH, 'classes.pkl')
    with open(classes_path, 'rb') as f:
        classes = pickle.load(f)
        no_classes = len(classes)

    #- Model - metric & confusion matrix paths
    METRICS_PATH = os.path.join('/'.join((args.weights).split('/')[:-2]), 'metrics')
    if not os.path.isdir(METRICS_PATH):
        os.mkdir(METRICS_PATH)
    CNF_MATRIX_PATH = os.path.join(METRICS_PATH, 'cnf_matrix_{}.png'.format(WEIGHTS))

    main(args)



    



