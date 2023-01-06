# Football Action Classification using CNN-LSTM

This document discusses about the methodologies employed in data preparation and model building processes for the classifying football events. We mainly focused on three events namely Throw-ins, Yellow cards & Corners.

## Dataset

SoccerNet[[1]](#References) is a large-scale dataset mainly created for soccer video understanding. Over the years, data is been labelled for various tasks such as action spotting, camera calibration, player re-identification and tracking. It is composed of 550 complete broadcast soccer games and 12 single camera games taken from the major European leagues. SoccerNet is not only dataset as it's maintainers also conduct yearly challenges where the best teams compete at the international level conferences.

We focused on dataset related to action-spotting task. It contains 17 actions or events are labelled with respective timestamps stored in `json` format for every match. The 17 labelled actions in the dataset are shown below,
|                    |                 | Actions          |                     |
|--------------------|-----------------|------------------|---------------------|
|      Penalty       |    Kick-off     |      Goal        |    Substitution     |
|      Offside       | Shots on target | Shots off target |     Clearance       |
|  Ball Out Of Play  |   **Throw-in**  |      Foul        | Indirect free-kick  |
|  Direct free-kick  |   **Corner**    | **Yellow card**  |     Red card        |
|  Yellow->red card  |                 |                  |                     |

Out of 17 actions, we focused on identifying three action namely Throw-in, Corner & Yellow card.

## Data statistics

We have manually watched some soccer matches and noted down the event or action time stamp & it's duration. Below statistics have been computed to get understanding on duration of the event or action,
|   Event or Action   | Duration (mean) | Duration (std. dev.) | Action clip (start) | Action clip (end) |  
|---------------------|-----------------|----------------------|---------------------|-------------------|
|      Yellow Card    |      0.909      |        0.437         |     TS + 0.0352     |    TS + 1.782     |
|      Red Card       |      0.909      |        0.437         |     TS + 0.0352     |    TS + 1.782     |
|      Throw-in       |      1.5        |        1.090         |     TS + 0.0694     |   TS + 2.130      |
|      Corner         |      1.1        |        0.515         |     TS - 0.6803     |   TS + 3.680      |

**Note**

- Duration(mean) & Duration (std. dev.) are in sec's
- `TS` means timestamp at which event or action occurs.
- For more information on event or action statistics, check it out [here](https://docs.google.com/spreadsheets/d/1MlLQifW1cku9VNuCqNe8ouMT92ChKka1_5lXDSMLB4Q/edit?usp=sharing).

From statistics, we got conclusion that the events we are trying to classify occupies a duration of 2-4 sec’s but we took 4 sec's as a standard. Also, we computed the start and end of the event/action clip which ranges from [label timestamp-1, label timestamp+3]. Finally, this ensures that event/action clip that captures most relevant information.

## Data preparation

We have created a handy script which uses ffmpeg to extract the action or event clips using the action spotting labels from the football match broadcast video.
Finally, all the extracted action or event clips are stored in the directory respective to their action and their match.

Total extracted action clips are over 500 but we've decided to have equal class distribution => Throw-in : Corner : Yellow-card = 120 : 120 : 120 clips.

## Data preprocessing

Each action video clip having resolution of `1024x720` & `4 sec's` at `25 FPS` produces `100 frames`. We took alternate frame (by sampling) which results to 50 images. Resizing(aspect ratio preserved) followed by normalization have been done to preprocess them.
The dataset distribution of train : valid : test =  81% : 9% : 10% is been employed.

## Model building

- Input video clip is sampled at 12 FPS (originally 24 FPS) i.e take 1 image for every 2 images.
- VGG-16 & Resnet-52 are used as feature extractors.
- The extracted features are being sent to LSTM model which contains 128 hidden units.
- Finally, we get three output class probabilities.

## Results

## Conclusion

## Future work

## Acknowledgements

## References

[1] &nbsp; Giancola, Silvio, Mohieddine Amine, Tarek Dghaily, and Bernard Ghanem. "Soccernet: A scalable dataset for action spotting in soccer videos." In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pp. 1711-1721. 2018.
