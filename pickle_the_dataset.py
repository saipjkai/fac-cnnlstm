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
#- FOR REPRODUCIBLE RESULTS ##### 


#- LIBS & FRAMEWORKS #####
import os 
import numpy as np
import cv2

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import pickle
#- LIBS & FRAMEWORKS #####


def get_data_paths(data_path, classes):
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
    
    print("We have {} videos".format(len(data_paths)))
    print("Videos distribution:\n{}".format(data_counter))
    print("Train+Validation(90%): {} videos & Test(10%): {}\n".format(.90*len(data_paths), .10*len(data_paths)))
    
    return data_paths, data_groundtruths


def preprocess(dims, data_paths, data_groundtruths=None, is_training_data=False):
    all_video_features = []
    for i in tqdm(range(len(data_paths))):
        cap = cv2.VideoCapture(data_paths[i])

        single_video_features = []
        while (True):
            read_success, current_frame = cap.read()
            
            if not read_success:
                break

            current_frame = cv2.resize(current_frame, (dims[2], dims[1]))
            current_frame = current_frame/255.0

            single_video_features.append(current_frame)

        cap.release()

        #- Sampling rate = 2 i.e 1 frame extracted per 2 frames
        single_video_features = single_video_features[:100:2]
        all_video_features.append(single_video_features)

    if is_training_data:
        all_video_features = np.array(all_video_features, dtype=np.float32)
        data_groundtruths = np.array(data_groundtruths, dtype=np.int32)
        return all_video_features, data_groundtruths
    else:
        all_video_features = np.array(all_video_features, dtype=np.float32)
        return all_video_features


def get_args():
    # args
    ap = ArgumentParser()
    ap.add_argument("--clips_dir", help='path to action video clips directory', required=True)
    # ap.add_argument("--dims", help="single unit data dimensions => (Depth, Height, Width, Channels)", default=(50, 128, 224, 3))
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    # args
    args = get_args()

    # root directory
    base_directory = os.path.abspath(".")
    
    # data - path, directories & processing 
    train_path = os.path.join(base_directory, args.clips_dir)
    
    # classes
    classes = {'Corner':0, 'Throw_in':1, 'Yellow_card':2}
    no_classes = len(classes)

    # data - single unit dimensions
    D = 50   # New Depth size => Number of frames.
    H = 128  # New Frame Height.
    W = 224  # New Frame Width.
    C = 3    # Number of channels.
    dims = (D, H, W, C) # Single Video shape.

    # data - processing
    data_paths, data_groundtruths = get_data_paths(train_path, classes)
    X, y = preprocess(dims, data_paths, data_groundtruths, True)

    # data - train & test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=True, random_state = seed_value)

    # data pickling
    data_pkl_path = os.path.join(base_directory, "data", "pickle")
    if not os.path.isdir(data_pkl_path):
        os.mkdir(data_pkl_path)
    
    # X, y
    X_pkl_path = os.path.join(data_pkl_path, 'Xtrain_{}x{}x{}x{}.pkl'.format(D, H, W, no_classes))
    y_pkl_path = os.path.join(data_pkl_path, 'ytrain_{}x{}x{}x{}.pkl'.format(D, H, W, no_classes))
    with open(X_pkl_path, 'wb') as X_train_pkl:
        pickle.dump(X_train, X_train_pkl)
    with open(y_pkl_path, 'wb') as y_train_pkl:
        pickle.dump(y_train, y_train_pkl)

    # Xtest, ytest
    X_test_path = os.path.join(data_pkl_path, 'Xtest_{}x{}x{}x{}.pkl'.format(D, H, W, no_classes))
    y_test_path = os.path.join(data_pkl_path, 'ytest_{}x{}x{}x{}.pkl'.format(D, H, W, no_classes))
    with open(X_test_path, 'wb') as X_test_pkl:
        pickle.dump(X_test, X_test_pkl)
    with open(y_test_path, 'wb') as y_test_pkl:
        pickle.dump(y_test, y_test_pkl)
