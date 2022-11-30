import os 
import numpy as np
import cv2

from tqdm import tqdm
import pickle

def get_test_data_paths(data_path):
    data_paths = []

    for file in tqdm(os.listdir(data_path)):
        vid_path = os.path.join(data_path, file)
        data_paths.append(vid_path)
    
    print("We have {} test videos".format(len(data_paths)))
    
    return data_paths


def get_train_data_paths(data_path, classes):
    data_paths = []
    data_groundtruths = []
    data_counter = dict()
    for label in classes:
        data_counter[label] = 0

    for class_name in classes:
        class_dir_path = os.path.join(data_path, class_name)
        for file in tqdm(os.listdir(class_dir_path)):
            vid_path = os.path.join(data_path, class_name, file)
            data_paths.append(vid_path)
            data_groundtruths.append(classes[class_name])
            data_counter[class_name] += 1
    
    print("We have {} training videos".format(len(data_paths)))
    print("Videos distribution:\n{}".format(data_counter))
    
    return data_paths, data_groundtruths


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
    # root directory
    base_directory = os.path.abspath(".")
    
    # data - path, directories & processing 
    train_path = os.path.join(base_directory, "data", "train")
    
    # classes
    classes = {'Corner':0, 'Throw_in':1, 'Yellow_card':2, 'Other':3}
    no_classes = len(classes)

    # data - single unit dimensions
    D = 50   # New Depth size => Number of frames.
    W = 128  # New Frame Width.
    H = 128  # New Frame Height.
    C = 3    # Number of channels.
    dims = (D, W, H, C) # Single Video shape.

    # data - processing
    data_paths, data_groundtruths = get_train_data_paths(train_path, classes)
    X, y = preprocess(dims, data_paths, data_groundtruths, True)

    # data pickling
    data_pkl_path = os.path.join(base_directory, "data", "pickle")
    if not os.path.isdir(data_pkl_path):
        os.mkdir(data_pkl_path)
    X_pkl_path = os.path.join(data_pkl_path, 'X_{}x{}_{}.pkl'.format(D, W, no_classes))
    y_pkl_path = os.path.join(data_pkl_path, 'y_{}x{}_{}.pkl'.format(D, W, no_classes))
    with open(X_pkl_path, 'wb') as X_pkl:
        pickle.dump(X, X_pkl)
    with open(y_pkl_path, 'wb') as y_pkl:
        pickle.dump(y, y_pkl)