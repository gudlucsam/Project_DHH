"""
https://github.com/FrederikSchorr/sign-language

For neural network training the method Keras.model.fit_generator is used. 
This requires a generator that reads and yields training data to the Keras engine.
"""


import glob
import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import keras

from frame import files2frames, images_normalize, frames_show

class FramesGenerator(tf.keras.utils.Sequence):
    """Read and yields video frames for Keras.model.fit_generator
    Generator can be used for multi-threading.

    Also encodes target text using one-hot encoder.

    Keyword arguments:
    sVideosPath -- (str) path to dataset of videos
    pickleDataPath -- (str) path to load/save processed frames in .npy data format
    labelsPath -- (str) path to load target text saved in .csv format
    nFrames -- (int) number of frames in video sequence
    nHeight -- (int) target height of processed video frames
    nWidth -- ((int) target height of processed video frames
    nChannels -- (int) channel of video frames (3)
    nBatchSize -- Batch of video data (default is 1 to process 1 video per batch)
    num_words -- (int) total number opf unique character tokens in target text to one-hot encode
    bShuffle -- (bool) whether to reshuffle dataset at start of epoch


    return generator that yields (x, y)
    """

    def __init__(self, sVideosPath, pickleDataPath, labelsPath, max_sentence_len=53, num_chars=44, nFrames=40, \
                 nHeight=224, nWidth=224, nChannels=3, nBatchSize=1, bShuffle = False):
        """
        Assume directory structure:
        ... / sPath / class / videoname / frames.jpg
        """

        'Initialization'
        self.nBatchSize = nBatchSize
        self.nFrames = nFrames
        self.nHeight = nHeight
        self.nWidth = nWidth
        self.nChannels = nChannels
        self.tuXshape = (nHeight, nWidth, nChannels)
        self.bShuffle = bShuffle

        # preprocess video frames and save to  pickleDataPath in npy format 
        videosDir2framesDir(sVideosPath, pickleDataPath, nTargetFrames=nFrames, nResizeMinDim=256, tuCropShape=(nHeight, nWidth))

        # encode labels
        print("encoding target texts...")
        self.targetLabels = self.textEncoder(labelsPath, max_sentence_len, num_chars)
        print("Done")

        # retrieve all videos frame from  pickleDataPath directories
        print("loading data from %s in .npy format...".format(pickleDataPath))
        self.dfVideos = np.load(pickleDataPath)
        print("Done")

        self.nSamples = self.dfVideos.shape[0]
        if self.nSamples == 0: raise ValueError("Found no frame directories files in " + pickleDataPath)
        print("Detected %d samples in %s ..." % (self.nSamples, pickleDataPath))
        
        self.on_epoch_end()

        return
    
    def textEncoder(self, labelsPath, max_sentence_len, num_chars):
        'Process target texts'
        df = pd.read_csv(labelsPath)
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
        encoded_sequences = tf.keras.utils.to_categorical(sequences, num_chars, dtype="int32")

        return np.array(encoded_sequences)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nSamples / self.nBatchSize))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nSamples)
        if self.bShuffle == True:
            np.random.shuffle(self.indexes)


    def __getitem__(self, nStep):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[nStep*self.nBatchSize:(nStep+1)*self.nBatchSize]
        # get batch of frames
        arX, arY = self.__data_generation(indexes)
        # labels already encoded
        return arX, arY


    def __data_generation(self, indexes):
        "Returns frames for 1 video, including normalizing & preprocessing"

        # retrive data using indexes
        arX = self.dfVideos[indexes]
        arY = self.targetLabels[indexes]

        return arX, arY

    def data_generation(self, seVideo):
        return self.__data_generation(seVideo)
