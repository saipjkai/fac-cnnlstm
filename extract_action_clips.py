from argparse import ArgumentParser
import os
from tqdm import tqdm
import json


def get_video_bitrate(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    bitrate = str(data["format"]["bit_rate"])
    b = (len(bitrate)-3)
    bitrate = int(bitrate[:b])
    bitrate = bitrate/1000

    return bitrate


def get_clip_start_end(start_time, span):
    mins, secs = list(map(int, start_time.split(":")))
    
    end_mins=mins
    end_secs = secs+span
    if end_secs > 59:
        end_secs = end_secs - 59
        end_mins = mins + 1

    start_mins=mins
    start_secs = secs-1
    if start_secs < 0:
        start_secs = start_secs+60
        start_mins = mins - 1
    
    start_mins = str(start_mins).zfill(2)
    start_secs = str(start_secs).zfill(2)
    end_mins = str(end_mins).zfill(2)
    end_secs = str(end_secs).zfill(2)

    return ("{}:{}".format(start_mins, start_secs), "{}:{}".format(end_mins, end_secs))


def main(args):
    # Input path
    ROOT_DIR = os.path.abspath(os.getcwd())
    matches_dir_path = os.path.abspath(os.path.join(os.getcwd(), args.in_dir))
    list_of_matches = os.listdir(matches_dir_path)

    # Store all clips in this directory 
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
        out_dir_path = os.path.abspath(os.path.join(os.getcwd(), args.out_dir))

    for match_dir in tqdm(list_of_matches):
        #- INPUT  --
        # single match - input path
        match_dir_path = os.path.abspath(os.path.join(matches_dir_path, match_dir))
        # match_dir_split = match_dir.split(" ")
        # updated_match_dir = "\ ".join(match_dir_split)

        # single match - videos 
        match_first_half_video_path = os.path.join(match_dir_path, "1_720p.mkv")
        match_second_half_video_path = os.path.join(match_dir_path, "2_720p.mkv")
        match_first_half_video_flag = False
        match_second_half_video_flag = False

        # single match labels
        try: 
            with open(os.path.join(match_dir_path, "Labels-v2.json"), "r") as json_file:
                labels = json.load(json_file)
                annotations = labels["annotations"]
        except:
            continue
            
        #- OUTPUT  --
        # single match - extracted clips path 
        out_match_path = os.path.join(out_dir_path, match_dir)
        if not os.path.isdir(out_match_path):
            os.mkdir(out_match_path)
        
        # single match - action paths
        for action in tqdm(args.actions):
            count=1
            action_clips_path = os.path.join(out_match_path, action)
            if not os.path.isdir(action_clips_path):
                os.mkdir(action_clips_path)
            
            for annotation in annotations:
                action_label = annotation["label"]
                action_gametime = annotation["gameTime"]
                action_visibility = annotation["visibility"]

                half_time, _, time_stamp = action_gametime.split(" ")

                if action == action_label and action_visibility == "visible":
                    action_clip_path = os.path.join(action_clips_path, "{}.mp4".format(count))
                    
                    start, end = get_clip_start_end(time_stamp, 3)

                    command_match_first_half_path = "\ ".join(match_first_half_video_path.split(" "))
                    command_match_second_half_path = "\ ".join(match_second_half_video_path.split(" "))
                    command_action_clip_path = "\ ".join(action_clip_path.split(" "))

                    if half_time == "1":
                        if not match_first_half_video_flag:
                            path_for_first_half_json_file = os.path.join(ROOT_DIR, 'dump1.json')
                            os.system("ffprobe -v quiet -print_format json -show_format -show_streams {} > {}".format(command_match_first_half_path, path_for_first_half_json_file))
                            match_first_half_video_flag = True

                        bit_rate = get_video_bitrate(path_for_first_half_json_file)

                        os.system("ffmpeg -ss 00:{} -to 00:{} -i {}  -b:v {}M {}".format(start, end, command_match_first_half_path, bit_rate, command_action_clip_path))
                        count+=1

                    elif half_time == "2":
                        if not match_second_half_video_flag:
                            path_for_second_half_json_file = os.path.join(ROOT_DIR, 'dump2.json')
                            os.system("ffprobe -v quiet -print_format json -show_format -show_streams {} > {}".format(command_match_first_half_path, path_for_second_half_json_file))
                            match_second_half_video_flag = False

                        bit_rate = get_video_bitrate(path_for_second_half_json_file)

                        os.system("ffmpeg -ss 00:{} -to 00:{} -i {}  -b:v {}M {}".format(start, end, command_match_first_half_path, bit_rate, command_action_clip_path))
                        count+=1
        
        # remove metadata 
        os.remove(path_for_first_half_json_file)
        os.remove(path_for_second_half_json_file)


def get_args():
    ap = ArgumentParser()
    ap.add_argument("--in_dir", help="Path to directory containing matches(each match - 1&2 halfs + Labels-v2.json", required=True)
    ap.add_argument("--actions", help="Give actions as list", default=["Corner", "Yellow card", "Throw-in"])
    ap.add_argument("--out_dir", help="Path to store action clips", default="output")
    args = ap.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    main(args)