"""
https://github.com/FrederikSchorr/sign-language

In some video classification NN architectures it may be necessary to calculate features 
from the (video) frames, that are afterwards used for NN training.

Eg in the MobileNet-LSTM architecture, the video frames are first fed into the MobileNet
and the resulting 1024 **features** saved to disc.
"""

import os
import glob
import time
import sys
import warnings

import numpy as np
import pandas as pd

import keras

from datagenerator import FramesGenerator


def features_2D_generator(sVideosPath, pickleDataPath, labelsPath, sFeatureBaseDir, keModel, nFramesNorm = 40):
    """Used by  InceptionV3 or MobileNet architecture to extract features from video frames.
    The (video) frames (2-dimensional) are fed into keModel (eg MobileNet/InceptionV3 without top layers)
    and the resulting features are save to sFeatureBaseDir.

    Features are used to train LSTM for sign translation

    
    Keyword arguments:
    sVideosPath -- (str) path to dataset of videos
    pickleDataPath -- (str) path to (train/test) data saved as np.array in .npy format
    labelsPath -- (str) path to target text saved in .csv format
    sFeatureBaseDir -- (str) path to save extracted CNN features (saved in npy format per video data)
    keModel -- (Keras Model) CNN model used as feature extractor (either MobileNet or InceptionV3)
    nFramesNorm -- (int) number of frames in sequence

    returns None
    """

    # do not (partially) overwrite existing feature directory
    if os.path.exists(sFeatureBaseDir): 
       warnings.warn("\nFeature folder " + sFeatureBaseDir + " already exists, calculation stopped") 
       return

    # create dir to save features
    os.makedirs(sFeatureBaseDir, exist_ok = True)

    # prepare frame generator - without shuffling!
    _, h, w, c = keModel.input_shape
    genFrames = FramesGenerator(sVideosPath, pickleDataPath, labelsPath, nFrames=nFramesNorm, nHeight=h, nWidth=w, nChannels=c, bShuffle=False)

    print("Predict features with %s ... " % keModel.name)
    nCount = 0
    # Predict - loop through all samples
    for i in range(genFrames.nSamples):

        # get data point and target 
        xFrames, yTarget = genFrames.data_generation(i)
        
        # create file path to save features
        sFeaturePath = sFeatureBaseDir + "/" + str(nCount)

        # check if already predicted
        if os.path.exists(sFeaturePath):
            print("Video %5d: features already written to %s" % (nCount, sFeaturePath))
            nCount += 1
            continue

        # get normalized frames and predict feature
        arFeature = keModel.predict(xFrames, verbose=0)

        # save to file
        np.savez(sFeaturePath, x = arFeature, y = yTarget)

        print("Video %5d: features %s saved to %s" % (nCount, str(arFeature.shape), sFeaturePath))
        nCount += 1

    print("%d features and target labels saved to files in %s" % (nCount, sFeatureBaseDir))

    return 