import random
import os

import numpy as np
from keras.utils import np_utils
from keras.utils import Sequence
from keras.utils import to_categorical
import cv2


def preprocess(X, y, num_classes, model_arch):
    X /=255.0
    if model_arch == "cnn_lstm":
        y = np.array(y)
    elif model_arch == "c3d":
        y = np_utils.to_categorical(np.array(y), num_classes)
    return X, y


def process_batch(X, y, dims, num_classes, batch_size, augmentation, model_arch, train):
    sample_size = dims[0]

    X_batch = np.zeros((batch_size, *dims), dtype='float32')
    if model_arch == "cnn_lstm":
        y_batch = []
    elif model_arch == "c3d":
        y_batch = np.zeros(batch_size, dtype='int')


    for batch_no in range(batch_size):
        # video
        video_path = X[batch_no]
        vc = cv2.VideoCapture(video_path)
        print(X[batch_no])
        print(y[batch_no])

        # augmentation
        if augmentation:
            # flip
            is_flip = random.randint(0, 1)

        # video to images, preprocessing & batch processing
        video_to_frames = []
        ret = True
        while ret:
            ret, frame = vc.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (dims[2], dims[1]))
                if train == True:
                    if augmentation:
                        frame = cv2.flip(frame, 1) if is_flip else frame
                    video_to_frames.append(frame)
                else:
                    video_to_frames.append(frame)
        vc.release()

        # processing for particular model architecture
        if model_arch == "cnn_lstm":
            video_to_frames = video_to_frames[0:100:2]
            y_batch.append(to_categorical(y[batch_no], num_classes=num_classes).astype('int32'))
        elif model_arch == "c3d":
            video_to_frames = video_to_frames[0:16]
            y_batch[batch_no] = y[batch]

        # inputs batch
        for count, img in enumerate(video_to_frames):
            X_batch[batch_no][count][:][:][:] = img

    return X_batch, y_batch


def batch_generator(X, y, num_classes, dims=(50, 128, 224, 3), batch_size=2, augmentation=False, model_arch="cnn_lstm", train=True):
    total_samples = len(X) 
    while True:
        Xy = list(zip(X, y))
        random.shuffle(Xy)

        X, y = zip(*Xy)

        for i in range(0, total_samples//batch_size):
            a = i*batch_size
            b = (i+1)*batch_size
            X_batch, y_batch = process_batch(X[a:b], y[a:b], dims, num_classes, batch_size, augmentation, model_arch, train)
            X_batch, y_batch = preprocess(X_batch, y_batch, num_classes, model_arch)
            
            yield X_batch, y_batch


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

            x_b.append(x_s)
            y_b.append(y_s)

        x_b = np.array(x_b, dtype=np.float32)
        y_b = np.array(y_b, dtype=np.int32)
        return x_b, y_b
    
    def postprocess(self, x_batch, y_batch):
        return x_batch, to_categorical(y_batch, num_classes=self.num_classes).astype(np.int32)
