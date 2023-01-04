import os
from matplotlib import pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


def plot_history(history, save_path):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.title('Model Performance for Video Action Detection', size=18, c="C7")
    plt.ylabel('Metrics', size=15, color='C7')
    plt.xlabel('Epoch No.', size=15, color='C7')

    plt.plot(history.history['accuracy'],  'o-', label='Training Data Accuracy', linewidth=2, c='C2') #C2 = green color.
    plt.plot(history.history['loss'],  '-', label='Training Loss', linewidth=2, c='C2') 
    plt.plot(history.history['val_accuracy'],  'o-', label='Validation Data Accuracy', linewidth=2, c='r') #r = red color.
    plt.plot(history.history['val_loss'],  '-', label='Validation Loss', linewidth=2, c='r') 
    
    plt.legend()    

    plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_cn_matrix(cnfMatrix, labels, save_path):
    disp = ConfusionMatrixDisplay(confusion_matrix=cnfMatrix, display_labels=labels)
    disp.plot()
    plt.savefig(save_path)
    plt.show()
