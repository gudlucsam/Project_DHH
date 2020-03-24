import os
import numpy as np

from utility_functions import target_text_encoder, process_videos
from encoder_decoder import lstm_models
from videocapture import predict_from_camera

# suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":

    # inceptionV3Params model parameters
    # model_params = {
    #     "mMame" : "inception",
    #     "input_shape": (299, 299, 3),
    #     "output_shape": (2048, )
    # }

    # mobilenet model parameters
    model_params = {
        "mName" : "mobilenet",
        "input_shape": (224, 224, 3),
        "output_shape": (1024,)
    }

    videos_path = "dataset"
    labels_path = "dataset_labels.csv"
    saved_model_path="saved_model/dhh.h5"

    nTargetFrames = 40
    latent_dim = 256
    nResizeMinDim = 256
    nFeatureLength = model_params["output_shape"][0]
    nHeight, nWidth, _ = model_params["input_shape"]
    tuCropShape = (nHeight, nWidth)
    # build encoder - decoder model
    instance = lstm_models(labels_path, model_params, nTargetFrames,
                            nFeatureLength, latent_dim=latent_dim,
                            saved_model_path=saved_model_path)

    # train model
    instance.train(videos_path, nResizeMinDim)
    # preprocess video frames
    frames = process_videos(videos_path, nTargetFrames=nTargetFrames,
                            nResizeMinDim=nResizeMinDim, tuCropShape=tuCropShape,
                            bRescale=True)
                            
    instance.predict(frames)

    # capture and predict from live webcam feed
    # predict_from_camera(instance, nTargetFrames, nHeight, nWidth)
