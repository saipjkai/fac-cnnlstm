# Football Action Classification using CNN-LSTM
This repository contains the code for data preparation, training and validation for the classifying Football events mainly focusing on Throw-ins, Yellow cards & Corners.

The network architecture is as follows,

- Input video clip is sampled at 12 FPS (originally 24 FPS) i.e take 1 image for every 2 images. 
- VGG-16 & Resnet-52 are used as feature extractors.
- The extracted features are being sent to LSTM model which contains 128 hidden units.
- Finally, we get three output class probabilities.

## Install requirements

```bash
$ pip install -r requiremets.txt
```

## 
