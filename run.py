from argparse import ArgumentParser
import sys
import os

import numpy as np
import cv2

from model_utils import load_model_from_weights

import datetime

from gtts import gTTS


def show_img(window_name, img):
    cv2.imshow(window_name, img)
    key = cv2.waitKey(40) & 0xFF
    if key == ord('p'):
        key = cv2.waitKey(0) & 0xFF
    elif key == ord('q'):
        exit(0)


def main(args):
    # run
    for match in matches:
        match_video_path = os.path.join(matches_path, match)
        
        for i in range(1, 3):
            match_video_half_path = os.path.join(match_video_path, "{}_720p.mkv".format(i))

            vc = cv2.VideoCapture(match_video_half_path)
            if not vc.isOpened():
                print("Unable to load video")
                exit(0)
            fps = vc.get(cv2.CAP_PROP_FPS)

            buffer_size=100
            frame_count=1
            frames = []
            ret = True
            while ret:
                ret, frame = vc.read()
                show_img("frame", frame)
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

                    if prediction_label is not None:
                        say_this = " ".join(prediction_label.split("_"))
                        # gtts_output = gTTS(text=say_this, lang=language, slow=False)
                        # gtts_output.save("trash.mp3")
                        # os.system("mpg321 trash.mp3")

                        print("{} - {} \t Event occurred: {}".format(timestamp_start, timestamp_end ,prediction_label))

                    frames = frames[-50:]

                frame_count += 1
            vc.close()

        break
    cv2.destroyAllWindows()


def get_args():
    # args
    ap = ArgumentParser()
    ap.add_argument("--weights", help='weights to load', required=True)
    ap.add_argument("--match_dir", help='path to directory containing entire matches', required=True)
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    # args 
    args = get_args()

    # base directory
    BASE_DIR = os.path.abspath(os.getcwd())

    # Google TTS text & language set
    say_this = ''  
    language = 'en'

    # Labels
    labels = ['Corner', 'Throw_in', 'Yellow_card']

    # model - load weights
    weights_path = os.path.join(BASE_DIR, args.weights)
    model = load_model_from_weights(weights_path)
    
    # Data directory 
    matches_path = os.path.join(BASE_DIR, args.match_dir)
    matches = os.listdir(matches_path)

    main(args)



    