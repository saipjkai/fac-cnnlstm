SEED_VALUE = 42


from argparse import ArgumentParser
import os 

from tqdm import tqdm

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pickle

#- is video valid? => based on dimensions
def check_video_dims(video_path, min_dim=112):
    vc = cv2.VideoCapture(video_path)
    width  = vc.get(cv2.CAP_PROP_FRAME_WIDTH)  
    height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT) 

    include_in_dataset = False
    if width >= min_dim and height >= min_dim:
        include_in_dataset = True
    return include_in_dataset


#- main
def main(args):
    # action classes 
    classes_list = os.listdir(INPUT_DIR)
    classes_list_sorted = sorted(classes_list)
    labels = dict()
    for num, class_label in enumerate(classes_list_sorted):
        labels[class_label] =  num

    # action class clips
    all_action_videos_paths = []
    action_labels = []
    for action in tqdm(classes_list_sorted):
        action_videos_path = os.path.join(INPUT_DIR, action)
        action_videos_list = os.listdir(action_videos_path)

        for video in tqdm(action_videos_list):
            video_path = os.path.join(action_videos_path, video)

            include_in_dataset = check_video_dims(video_path)
            # for training and testing purposes
            if include_in_dataset:
                all_action_videos_paths.append(os.path.abspath(video_path))
                action_labels.append(labels[action])
    
    X_train, X_test, y_train, y_test = train_test_split(all_action_videos_paths, action_labels, test_size=0.1, random_state=SEED_VALUE, stratify=action_labels)

    with open(os.path.join(OUTPUT_DIR, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X_train, f)
    with open(os.path.join(OUTPUT_DIR, 'X_test.pkl'), 'wb') as f:
        pickle.dump(X_test, f)
    with open(os.path.join(OUTPUT_DIR, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(OUTPUT_DIR, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    with open(os.path.join(OUTPUT_DIR, 'classes.pkl'), 'wb') as f:
        pickle.dump(list(labels.keys()), f)


#- args
def get_args():
    ap = ArgumentParser()
    ap.add_argument("--in_dir", help="Path to directory containing action directories with respective videos", required=True)
    ap.add_argument("--out_dir", help="Output pickle directories", required=True)
    
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    # args
    args = get_args()
    
    # input & output directories
    BASE_DIR = os.getcwd()
    INPUT_DIR = os.path.abspath(args.in_dir)
    OUTPUT_DIR = os.path.abspath(args.out_dir)
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # main
    main(args)    




