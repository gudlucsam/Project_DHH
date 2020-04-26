"""
https://github.com/FrederikSchorr/sign-language

Extract frames from a video (or many videos). 
Plus some frame=image manipulation utilities.
"""

import os
import glob
import warnings
import random
from subprocess import check_output
from keras.preprocessing.text import Tokenizer

import numpy as np
import pandas as pd

import cv2


"""
Utilities for extraction of frames from a video (or videos);
Plus manipulation of frames.

@references: https://github.com/FrederikSchorr/sign-language
"""

def image_resize_aspectratio(arImage, nMinDim = 256):
    """
    Resize aspect ratio of image.
    Rescale height to 256.

    Keyword arguments:
    arImage -- np.array
    nMinDim -- Minimum Dimension (default 256)

    returns np.array of floats
    """
    if (len(arImage.shape) < 2 ): raise ValueError("Image doesnot exist")
    nHeigth, nWidth, _ = arImage.shape

    if nWidth >= nHeigth:
        # wider than high => map heigth to 224
        fRatio = nMinDim / nHeigth
    else: 
        fRatio = nMinDim / nWidth

    if fRatio != 1.0:
        arImage = cv2.resize(arImage, dsize = (0,0), fx = fRatio, fy = fRatio, interpolation=cv2.INTER_LINEAR)

    return arImage


def images_resize_aspectratio(arImages, nMinDim = 256):
    """
    Resize aspect ratio of images.

    Keyword arguments:
    arImages -- List of np.array
    nMinDim -- Minimum Dimension (default 256)

    returns np.array of floats
    """
    nImages, _, _, _ = arImages.shape
    liImages = []
    for i in range(nImages):
        arImage = image_resize_aspectratio(arImages[i, ...])
        liImages.append(arImage)

    return np.array(liImages)


def target_text_encoder(labels_path):
    # Vectorize the data.
    target_texts = []
    target_characters = set()
    # read target text data
    df = pd.read_csv(labels_path)
    target_samples = df['translation'].values.tolist()
    for target_text in target_samples:
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        target_texts.append(target_text)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    target_characters = sorted(list(target_characters))
    num_decoder_tokens = len(target_characters)
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    # print('Number of samples:', len(target_texts))
    # print('Number of unique output tokens:', num_decoder_tokens)
    # print('Max sequence length for outputs:', max_decoder_seq_length)

    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])
    reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

    decoder_input_data = np.zeros(
        (len(target_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(target_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for i, target_text in enumerate(target_texts):
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1
        decoder_input_data[i, t + 1:, target_token_index[' ']] = 1
        decoder_target_data[i, t:, target_token_index[' ']] = 1

    return (max_decoder_seq_length, num_decoder_tokens, target_token_index, \
            reverse_target_char_index, decoder_input_data, decoder_target_data)

def video2frames(sVideoPath, nResizeMinDim):
    """
    Read video file with OpenCV and return array of frames
    The frame rate depends on the video (and cannot be set)

    if nResizeMinDim != None: Frames are resized preserving aspect ratio 
    so that the smallest dimension is eg 256 pixels, with bilinear interpolation

    Keyword arguments:
    sVideoPath -- path to video file
    nMinDim -- minimum dimension image must have

    returns np.array of floats.
    """
    
    # Create a VideoCapture object and read from input file
    oVideo = cv2.VideoCapture(sVideoPath)
    if (oVideo.isOpened() == False): raise ValueError("Error opening video file")

    lstOfFrames = []

    # Read until video is completed
    while(True):
        
        (bGrabbed, arFrame) = oVideo.read()
        if bGrabbed == False: break

        if nResizeMinDim != None:
            # resize image
            arFrameResized = image_resize_aspectratio(arFrame, nResizeMinDim)
        
		# Save the resulting frame to list
        lstOfFrames.append(arFrameResized)
   
    return np.array(lstOfFrames, dtype=np.float32)


def frames2files(arFrames, sTargetDir):
    """
    Write array of frames to jpg files.
    Keyword arguments:
    arFrames -- np.array of shape: (number of frames, height, width, depth)
    sTargetDir -- dir to hold jpg files

    returns None
    """
    os.makedirs(sTargetDir, exist_ok=True)
    for nFrame in range(arFrames.shape[0]):
        cv2.imwrite(sTargetDir + "/frame%04d.jpg" % nFrame, arFrames[nFrame, :, :, :])

    return


def files2frames(sPath):
    """
    Read jpg files to array of frames.
    Keyword arguments:
    sPath -- dir path to image files

    returns None
    """
    # important to sort image files upfront
    liFiles = sorted(glob.glob(sPath + "/*.jpg"))
    if len(liFiles) == 0: raise ValueError("No frames found in " + sPath)

    liFrames = []
    # loop through frames
    for sFramePath in liFiles:
        arFrame = cv2.imread(sFramePath)
        liFrames.append(arFrame)

    return np.array(liFrames, dtype = np.float32)
    
    
def frames_downsample(arFrames, nFramesTarget):
    """
    Adjust number of frames (eg 123) to nFramesTarget (eg 79)
    works also if originally less frames then nFramesTarget

    Keyword arguments:
    arFrames -- number of frames
    nFramesTarget -- target number of frames.

    Returns np.array of floats
    """

    nSamples, _, _, _ = arFrames.shape
    if nSamples == nFramesTarget: return arFrames

    # down/upsample the list of frames
    fraction = nSamples / nFramesTarget
    index = [int(fraction * i) for i in range(nFramesTarget)]
    lstOfTarget = [arFrames[i,:,:,:] for i in index]
    # print("Change number of frames from %d to %d" % (nSamples, nFramesTarget))

    return np.array(lstOfTarget)
    
    
def image_crop(arFrame, nHeightTarget, nWidthTarget):
    """
    Crop a frame to specified size, choose centered image

    Keyword arguments:
    arFrame -- np.array representing image
    nHeightTarget -- height of target image
    nWidthTarget -- width of target image

    Returns np.array of floats
    """
    nHeight, nWidth, _ = arFrame.shape

    if (nHeight < nHeightTarget) or (nWidth < nWidthTarget):
        raise ValueError("Image height/width too small to crop to target size")

    # calc left upper corner
    sX = int(nWidth/2 - nWidthTarget/2)
    sY = int(nHeight/2 - nHeightTarget/2)

    arFrame = arFrame[sY:sY+nHeightTarget, sX:sX+nWidthTarget, :]

    return arFrame


def images_crop(arFrames, nHeightTarget, nWidthTarget):
    """
    Crop array of frames to specified size, choose centered image

    Keyword arguments:
    arFrames -- np.array representing list of images, shape : (nSamples, nHeight, nWidth, nDepth)
    nHeightTarget -- height of target image
    nWidthTarget -- width of target image

    Returns np.array of floats
    """
    nSamples, nHeight, nWidth, nDepth = arFrames.shape

    if (nHeight < nHeightTarget) or (nWidth < nWidthTarget):
        raise ValueError("Image height/width too small to crop to target size")

    # calc left upper corner
    sX = int(nWidth/2 - nWidthTarget/2)
    sY = int(nHeight/2 - nHeightTarget/2)

    arFrames = arFrames[:, sY:sY+nHeightTarget, sX:sX+nWidthTarget, :]

    return arFrames


def images_rescale(arFrames):
    """
    Rescale array of images (rgb 0-255) to [-1.0, 1.0]

    Keyword arguments:
    arFrames -- np.array of images

    Returns np.array of floats
    """
    ar_fFrames = arFrames /  127.5
    ar_fFrames -= 1.0

    return arFrames

def save_to_npy(sFrames, sPath):
    """
    Write np.array to file

    """
    dir_path = "/gdrive/My Drive/Capstone/frames_npy/"
    # if file exist do not save
    if not os.path.exists(dir_path): 
        os.makedirs(dir_path, exist_ok = True)
    
    # create full path to save frames
    sPath = os.path.join(dir_path, sPath)
    if os.path.exists(sPath):
        warnings.warn("\nFrames already saved to " + sPath) 
        return

    # save frames to npy
    np.save(sPath, sFrames)
    
    return 


def images_normalize(arFrames, nTargetFrames, nHeight, nWidth, bRescale = True):
    """ Several image normalizations/preprocessing: 
        - downsample number of frames
        - crop to centered image
        - rescale rgb 0-255 value to [-1.0, 1.0] - only if bRescale == True

    Keyword arguments:
    arFrames -- np.array representing images
    nTargetFrames -- target number of images after downsampling
    nHeight -- height of target image after cropping
    nWidth -- width of target image after cropping
    nRescale -- indicates whether to rescale px values of images

    Returns np.array of floats
    """

    # normalize the number of frames (assuming typically downsampling)
    arFrames = frames_downsample(arFrames, nTargetFrames)
    # crop to centered image
    arFrames = images_crop(arFrames, nHeight, nWidth)
    if bRescale:
        # normalize to [-1.0, 1.0]
        arFrames = images_rescale(arFrames)
    else:
        if np.max(np.abs(arFrames)) > 1.0: warnings.warn("Images not normalized")

    return arFrames


def frames_show(arFrames, nWaitMilliSec = 100):
    """
    Utility func to visualize images
    
    Keyword arguments:
    arFrames -- np.array representing images
    nWaitMillisec -- sec to wait before destroying image window
    """

    nFrames, nHeight, nWidth, nDepth = arFrames.shape
    
    for i in range(nFrames):
        cv2.imshow(arFrames[i, :, :, :])
        cv2.waitKey(nWaitMilliSec)

    return


def video_length(sVideoPath):
    """
    Uses mediainfo to obtain video info

    Keyword arguments:
    sVideoPath -- path to video file
    """

    cap = cv2.VideoCapture(sVideoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    return duration


def process_videos(sVideoDir, nTargetFrames = None, 
    nResizeMinDim = None, tuCropShape = None, bRescale=True):
    """
    Extract frames from videos, and apply preprocessing
    Input video structure:
    ... dataset / video.mpeg

    Keyword arguments: 
    sVideoDir -- dir of videos
    nTargetFrames -- number of frames after downsampling
    nResizeminDim -- minimum dimension after resizing
    tuCropShape -- tuple (nHeight, nWidth) to crop image - we unpack with * operator
    
    bRescale -- rescale rgb 0-255 value to [-1.0, 1.0] if True (default True)

    preprocess and save preprocessed data to npy format in 
    """

    # initialize list 
    sTragetFrames = []
    # get videos. Assume .../dataset / video.mpg
    dfVideos = pd.DataFrame(sorted(glob.glob(sVideoDir + "/*.[mp4][mpg]*")), columns=["sVideoPath"])
    print("Located {} videos in {}, extracting and processing frames...".format(len(dfVideos), sVideoDir))
    if len(dfVideos) == 0: raise ValueError("No videos found")

    # nCounter = 0
    # loop through all videos and extract frames
    for i, sVideoPath in enumerate(dfVideos.sVideoPath):
        print("processing video", i)
        # slice videos into frames with OpenCV
        arFrames = video2frames(sVideoPath, nResizeMinDim)

        # length and fps
        # fVideoSec = video_length(sVideoPath)
        # nFrames = len(arFrames)
        # fFPS = nFrames / fVideoSec   

        # preprocess images
        arFrames = images_normalize(arFrames, nTargetFrames, *tuCropShape, bRescale)
        
        # append frames to np.array
        sTragetFrames.append(arFrames)


        # print("Video %5d | %5.1f sec | %d frames | %4.1f fps | processed %s frames" % (nCounter, fVideoSec, nFrames, fFPS, str(arFrames.shape)))
        # nCounter += 1

    sTragetFrames = np.array(sTragetFrames, dtype="float32")     

    return sTragetFrames
                    