import os
import glob
import time
import sys
import warnings

import keras
import numpy as np
import pandas as pd

from datagenerator import data_generator


def features_generator(videos_path, labels_path, keModel, max_sentence_len=53, \
                            num_chars=44, nTargetFrames=40, nResizeMinDim=256):
    """
    Used by  InceptionV3 or MobileNet architecture to extract features from video frames.
    The (video) frames (2-dimensional) are fed into keModel (eg MobileNet/InceptionV3 without top layers)
    and the resulting features are save to sFeatureBaseDir.
    Features are used to train LSTM for sign translation

    
    Keyword arguments:
    sVideosPath -- (str) path to dataset of videos
    labelsPath -- (str) path to target text saved in .csv format
    keModel -- (Keras Model) CNN model used as feature extractor (either MobileNet or InceptionV3)
    nTargetFrames -- (int) number of frames in sequence

    returns None
    """
    # prepare frame generator - without shuffling!
    _, h, w, _ = keModel.input_shape
    tuCropShape = (h, w)
    frames, labels = data_generator(videos_path, labels_path, tuCropShape=tuCropShape,
                                    max_sentence_len=max_sentence_len, 
                                    num_chars=num_chars, nTargetFrames=nTargetFrames,
                                    nResizeMinDim=nResizeMinDim)

    print("Predict features with %s ... " % keModel.name)
    nSamples = frames.shape[0]
    feature_frames = []
    # Predict - loop through all samples
    for i in range(nSamples):
        print("extracting features for frames from video ", i)
        # get data point and target 
        xFrames = frames[i]
        # get normalized frames and predict feature
        arFeature = keModel.predict(xFrames, verbose=0)
        # append features
        feature_frames.append(arFeature)

    return np.array(feature_frames), labels