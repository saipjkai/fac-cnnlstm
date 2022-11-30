from argparse import ArgumentParser
import os
from shutil import copy

def get_args():
    ap = ArgumentParser()
    ap.add_argument("--i", help="Input path", required=True)

    args = ap.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    # Input
    IN_DIR_PATH = os.path.abspath(args.i) 
    MATCHES_LIST = os.listdir(IN_DIR_PATH)
    MATCHES_COUNT = len(MATCHES_LIST)

    # Output
    ACTIONS = ['Corner', 'Throw-in', 'Yellow card']
    OUT_DIR_PATH = os.path.abspath(os.path.join(os.getcwd(), 'final_action_clips'))
    if not os.path.isdir(OUT_DIR_PATH):
        os.mkdir(OUT_DIR_PATH)
        for action in ACTIONS:
            os.mkdir(os.path.join(OUT_DIR_PATH, action))

    # Data stats
    final_action_clips_count = dict()
    for action in ACTIONS:
        final_action_clips_count[action] = 1
    
    # Combining data
    for count, match in enumerate(MATCHES_LIST):
        match_path = os.path.join(IN_DIR_PATH, match)
        for action in ACTIONS:
            match_action_path = os.path.join(match_path, action)
            if os.path.isdir(match_action_path):
                match_action_clips_count = len(os.listdir(match_action_path))
                sorted_match_action_clips = ['{}.mp4'.format(i) for i in range(1, match_action_clips_count+1)]
                for action_clip in sorted_match_action_clips:
                    copy(os.path.join(match_path, action, action_clip), os.path.join(OUT_DIR_PATH, action, "{}.mp4".format(final_action_clips_count[action])))
                    final_action_clips_count[action] += 1
        
        print("match: {} Completed \t => {:.2%}".format(match, ((count+1)/MATCHES_COUNT)))        

