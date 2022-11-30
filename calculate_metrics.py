import pickle

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load actual & predictions
    with open('y_actual.pkl', 'rb') as act_pkl:
        y_test = pickle.load(act_pkl)
    
    with open('y_pred_prob.pkl', 'rb') as pred_pkl:
        y_proba = pickle.load(pred_pkl)

    y_pred = []
    for i in y_proba:
        y_pred.append(np.argmax(i))

    # Class Labels
    labels = ['Corner', 'Throw_in', 'Yellow_card', 'Other']

    # Generate generalization metrics
    # Accuracy
    # accuracy = accuracy_score(y_test, y_pred)
    
    # AUC score
    auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr')

    # Confusion matrix

    cn_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cn_matrix, display_labels=labels)
    
    # print("Accuracy: {}".format(accuracy))
    print("AUC score: {}".format(auc_score))
    
    disp.plot()
    plt.savefig('30112022_v8_metrics.png')
    plt.show()