"""
https://github.com/FrederikSchorr/sign-language

For neural network training the method Keras.model.fit_generator is used. 
This requires a generator that reads and yields training data to the Keras engine.
"""


import glob
import os
import sys
import keras

import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from utility_functions import process_videos, frames_show
from keras.utils import Sequence, to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences




def textEncoder(labels_path, max_sentence_len, num_chars):
    """
    Encode target texts
    """

    df = pd.read_csv(labels_path)
    samples = df['translation'].values.tolist()

    target_samples = []
    for text in samples:
        # add "\t" and "\n" to depict start and end of sentence respectively
        text = "\t"+text+"\n"
        target_samples.append(text)
    
    # one-hot encode by character level
    tokenizer = Tokenizer(num_words=num_chars, char_level=True)
    tokenizer.fit_on_texts(target_samples)
    sequences = tokenizer.texts_to_sequences(target_samples)
    sequences = pad_sequences(sequences, maxlen=max_sentence_len, padding="post")
    encoded_sequences = to_categorical(sequences, num_chars, dtype="int32")

    return np.array(encoded_sequences)


def data_generator(videos_path, labels_path, tuCropShape=(224, 224), max_sentence_len=53, \
                    num_chars=44, nTargetFrames=40, nResizeMinDim=256):
    
    # preprocess video frames
    frames = process_videos(videos_path, nTargetFrames=nTargetFrames,
                            nResizeMinDim=nResizeMinDim, tuCropShape=tuCropShape,
                            bRescale=True)

    # preprocess target labels 
    labels = textEncoder(labels_path, max_sentence_len, num_chars)
    
    return frames, labels



# if __name__ == "__main__":
#     videos_path = "dataset"
#     labels_path = "dataset_labels.csv"
#     frames, labels = data_generator(videos_path, labels_path, tuCropShape=(224, 224),
#                                     max_sentence_len=53, num_chars=44, nTargetFrames=40,
#                                     nResizeMinDim=256)

#     print(frames.shape, labels.shape)
