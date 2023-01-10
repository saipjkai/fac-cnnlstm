from argparse import ArgumentParser
import sys
import os

import numpy as np
import cv2

from model_utils import load_model_from_weights

import datetime


def show_img(window_name, img):
    cv2.imshow(window_name, img)
    key = cv2.waitKey(40) & 0xFF
    if key == ord('p'):
        key = cv2.waitKey(0) & 0xFF
    elif key == ord('q'):
        exit(0)

# main
def main(args):
    vc = cv2.VideoCapture(args.video)
    if not vc.isOpened():
        print("Unable to load video")
        exit(0)
    fps = vc.get(cv2.CAP_PROP_FPS)

    buffer_size=100
    frame_count=1
    frames = []
    ret = True
    output_string = None
    while ret:
        ret, frame = vc.read()
        vis_frame = frame.copy()

        frame_resized = cv2.resize(frame, (224, 128))
        frame_resized = frame_resized/255.0
        frames.append(frame_resized)

        # show_img("frame", frame)
        if len(frames) % buffer_size == 0:
            video_clip = frames[::2]
            video_clip_np = np.array(video_clip)
            
            prediction = model.predict(video_clip_np.reshape(-1, buffer_size//2, 128, 224, 3))
            max_prob_index  = np.argmax(prediction[0])
            max_action_prob = prediction[0][max_prob_index]
            if max_action_prob > .67:
                prediction_label = labels[max_prob_index]
            else:
                prediction_label = None

            timestamp_start = datetime.timedelta(seconds=(frame_count-buffer_size)/fps) 
            timestamp_end = datetime.timedelta(seconds=(frame_count)/fps) 

            # output_string = "{} - {}, Event occurred: {}".format(timestamp_start, timestamp_end, prediction_label)
            output_string = "Event occurred: {} in last 4 sec's".format(prediction_label)
            if prediction_label is not None:
                print(output_string)
            frames = frames[-50:]

        if output_string is not None:
            cv2.putText(vis_frame, output_string, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        show_img("frame", vis_frame)
        vc_out.write(vis_frame)
        frame_count += 1
    
    vc.close()
    vc_out.release()
    cv2.destroyAllWindows()


def get_args():
    # args
    ap = ArgumentParser()
    ap.add_argument("--weights", help='weights to load', required=True)
    ap.add_argument("--video", help='path to broadcast soccer match', required=True)
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    # args 
    args = get_args()

    # base directory
    BASE_DIR = os.path.abspath(os.getcwd())

    # Labels
    labels = ['Corner', 'Throw_in', 'Yellow_card']

    # model - load weights
    weights_path = os.path.join(BASE_DIR, args.weights)
    model = load_model_from_weights(weights_path)

    # save output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vc_out = cv2.VideoWriter('result.mp4', fourcc, 24.0, (1280,720))

    main(args)



    