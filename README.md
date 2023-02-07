# Football Action Classification 
This repository contains the code for data preparation, training and validation for the classifying Football events mainly focusing on Throw-ins, Yellow cards & Corners.

We have implemented two different methodologies to approach the football action classification namely,
1. CNN - LSTM
2. C3D

## Install requirements

```bash
$ pip install -r requirements.txt
```

## Data preparation

- SoccerNetv2 `action-spotting` dataset is used for model building. Sign the NDA [here](https://silviogiancola.github.io/SoccerNetv2/) and get the credentials to download football broadcast videos. The labels for `action-spotting` dataset can be downloaded using SoccerNet api found [here](https://www.soccer-net.org/data).  

- After downloading the football broadcast videos and their respective labels create the following directory structure,
    ```bash
    matches/
    ├── 2016-08-20 - 19-00 AS Roma 4 - 0 Udinese
    │   ├── 1_720p.mkv
    │   ├── 2_720p.mkv
    │   └── Labels-v2.json
    ├── 2016-08-21 - 21-45 Pescara 2 - 2 Napoli
    │   ├── 1_720p.mkv
    │   ├── 2_720p.mkv
    │   └── Labels-v2.json
    ├── ..
    ..
    ```

- We can extract clips using `extract_action_clips.py`,
    ```bash
    $ python extract_action_clips.py --in_dir [/path/to/directory/structure/shown/above/] --out_dir [/path/to/store/video/clips/]
    ```
   The extracted clips are created in `out_dir` directory path and with the clips in their respective action named directories and structure is as follows,
    ```bash
    output/
    ├── 2016-08-20 - 19-00 AS Roma 4 - 0 Udinese
    │   ├── Corner
    │   ├── Throw-in
    │   └── Yellow card
    ├── 2016-08-21 - 21-45 Pescara 2 - 2 Napoli
    │   ├── Corner
    │   ├── Throw-in
    │   └── Yellow card
    ..
    ```

- We can combine all the extracted clips using `combine_clips_and_rename.py` as follows,
    ```bash
    $ python combine_clips_and_rename.py --i [/path/to/extracted/video/clips]
    ```
    Now, all the extracted clips of all matches will come into single directory i.e `final_action_clips` with their respective action directories inside it,
    ```bash
    final_action_clips/
    ├── Corner
    │   ├── 1.mp4
    │   ├── 2.mp4
    │   ├── 3.mp4
    │   ..
    ├── Throw_in   
    │   ├── 1.mp4
    │   ├── 2.mp4
    │   ├── 3.mp4
    │   ..
    └── Yellow_card
        ├── 1.mp4
        ├── 2.mp4
        ├── 3.mp4
        ..   
    ```
    These clips can be directly used for training purposes.

- For the ease of training and testing, we store the video paths and given class in the form of `pickle` format.
    ```
    $ python pickle_the_dataset.py --in_dir [/path/to/final_action_clips/] --out_dir [/path/to/store/pkls]
    ```

## Training

- We can train our model by running the `main.py`,
    ```bash
    $ python main.py --pkl_dir [/path/to/directory/containing/pkls] --version 1 
    ```
    - *`--version` represents version number and after entire training best weights and training curves are stored with current date and version number in `backup` directory.*
    - data preparation and training scripts can be changed to either approaches easily in source code.

## Metrics

- For calculating the metrics on test data `pickle` with the trained model, we can run `run_metrics.py`
    ```bash
    $ python run_metrics.py --weights backup/2023-01-08/weights/weights_2023-01-08_v1_tf.h5 --test_path [/path/to/directory/containing/pkls]
    ```
    This will print the metrics such as `AUC`, `confusion matrix` and `accuracy`.

## Model inference on entire match

- We can create a directory containing matches as below structure as follows, 
    ```bash
    videos/
    └── 2016-09-10 - 19-30 RB Leipzig 1 - 0 Dortmund
    │   ├── 1_720p.mkv
    │   └── 2_720p.mkv
    ..
    ```
    And then we can run the trained model on entire matches in the directory using `run.py` and watch the actions predicted on terminal output.
    ```bash
    $ python run.py --weights backup/2023-01-08/weights/weights_2023-01-08_v1_tf.h5 --match_dir [/path/to/directory/containing/matches]
    ```

## Additional details
- For in-depth documentation, please have a look at [Documentation.md](./Documentation.md)
- For presentation slides, please have a look at [Football Action Classification-Capability.pptx](./Football%20Action%20Classification%20-%20Capability.pptx)

