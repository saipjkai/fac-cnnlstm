import random
import os

import numpy as np
from keras.utils import Sequence
from keras.utils import to_categorical
import cv2



class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, dims=(50, 128, 224, 3), model_arch="cnn_lstm", num_classes=3):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.dims = dims
        self.model_arch = model_arch
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # preprocess   
        x_batch, y_batch = self.preprocess(batch_x, batch_y)

        # postprocess
        x_batch, y_batch = self.postprocess(x_batch, y_batch)

        return x_batch, y_batch

    def preprocess(self, x_batch, y_batch):
        x_b = []
        y_b = []
    
        for x_, y_ in zip(x_batch, y_batch):
            vc = cv2.VideoCapture(x_)

            x_s = []
            while (True):
                read_success, frame = vc.read()
                if not read_success:
                    break

                frame = cv2.resize(frame, (self.dims[2], self.dims[1]))
                frame = frame/255.0

                x_s.append(frame)

            vc.release()

            if self.model_arch == "cnn_lstm":
                x_s = x_s[:100:2]
                y_s = y_
            elif self.model_arch == "c3d":
                x_s = x_s[:16]
                y_s = y_

            x_b.append(x_s)
            y_b.append(y_s)

        x_b = np.array(x_b, dtype=np.float32)
        y_b = np.array(y_b, dtype=np.int32)
        return x_b, y_b
    
    def postprocess(self, x_batch, y_batch):
        return x_batch, to_categorical(y_batch, num_classes=self.num_classes).astype(np.int32)
