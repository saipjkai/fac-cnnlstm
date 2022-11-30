import os
import numpy as np
import cv2

import keras
import pickle

from tqdm import tqdm

def get_test_data_paths(data_path):
    data_paths = []

    for file in tqdm(os.listdir(data_path)):
        vid_path = os.path.join(data_path, file)
        data_paths.append(vid_path)
    
    print("We have {} test videos".format(len(data_paths)))
    
    return data_paths


def preprocess(dims, data_paths, data_groundtruths=None, is_training_data=False):
    all_video_features = []
    for i in tqdm(range(len(data_paths))):
        cap = cv2.VideoCapture(data_paths[i])

        single_video_frames = []
        while (True):
            read_success, current_frame = cap.read()
            
            if not read_success:
                break

            current_frame = cv2.resize(current_frame, (dims[1], dims[2]))
            single_video_frames.append(current_frame)

        cap.release()

        #- Direct Sequential Frames
        single_video_direct_sequential_frames = single_video_frames[25:25+dims[0]]
        single_video_direct_sequential_frames = np.array(single_video_direct_sequential_frames)
        single_video_direct_sequential_frames.resize(dims)

        all_video_features.append(single_video_direct_sequential_frames)
    
    all_video_features = np.array(all_video_features)
    
    if is_training_data:
        data_groundtruths = np.array(data_groundtruths, dtype=np.int32)
        return all_video_features, data_groundtruths
    else:
        return all_video_features


if __name__ == "__main__":
    # data - single unit dimensions
    D = 50   # New Depth size => Number of frames.
    W = 128  # New Frame Width.
    H = 128  # New Frame Height.
    C = 3    # Number of channels.
    dims = (D, W, H, C) # Single Video shape.

    # base dir
    base_directory = os.path.abspath(os.getcwd())
    # test data
    test_path  = os.path.join(base_directory, "data", "test")

    test_data_paths = get_test_data_paths(test_path)
    X_test = preprocess(dims, test_data_paths)

    # Model - load weights
    weights_path = os.path.join(base_directory, "models", "weights_2022-11-30_v8_tf.h5")
    model = keras.models.load_model(weights_path)

    # Y Test - Actual & Predictions 
    y_test = []
    y_preds = []
    test_data_dist = {'Corner': 0, 'Throw_in': 0, 'Yellow_card': 0, 'Other': 0}
    for i, j in zip(X_test, test_data_paths):
        # test file name
        test_data_file_name = j.split('/')[-1]
        if test_data_file_name[0] == 'c':
            y_test.append(0)
            test_data_dist['Corner'] += 1
        elif test_data_file_name[0] == 't':
            y_test.append(1)
            test_data_dist['Throw_in'] += 1
        elif test_data_file_name[0] == 'y':
            y_test.append(2)
            test_data_dist['Yellow_card'] += 1
        elif test_data_file_name[0] == 'o':
            y_test.append(3)
            test_data_dist['Other'] += 1

        # test predictions
        y_pred = model.predict(i.reshape(-1, D, W, H, C))
        y_preds.append(y_pred[0].tolist())

    # Saving predictions & actual for evaluting metrics
    with open('y_pred_prob.pkl', 'wb') as pred_pkl:
        pickle.dump(y_preds, pred_pkl)
    with open('y_actual.pkl', 'wb') as act_pkl:
        pickle.dump(y_test, act_pkl)

    print("Test data distribution: {}".format(test_data_dist))