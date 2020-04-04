import os
import numpy as np

from model_lib import config

from model_lib.utility_functions import target_text_encoder, process_videos
from model_lib.encoder_decoder import lstm_models
from model_lib.videocapture import predict_from_camera

from flask import Flask, jsonify

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

# instantiate flask
app = Flask(__name__)

model = None



@app.route("/")
def index():
    return "Hello Word"

@app.route("/select_model", methods=['GET'])
def select_feature_extraction_model():

    # retrieve payload
    payload = None

    # init response
    res = {"success": False}

    # make model instance global
    global model

    # select cnn model to use
    if payload == "inception":
        model_params = config.inception_model_params
    else:
        model_params = config.mobilenet_model_params

    # retrieve model params
    nFeatureLength = model_params["output_shape"][0]
    saved_model_path = model_params["saved_model_path"]

    labels_path = config.labels_path
    nTargetFrames = config.nTargetFrames
    latent_dim = config.latent_dim


    # build encoder - decoder model
    model = lstm_models(labels_path, model_params, nTargetFrames,
                            nFeatureLength, latent_dim=latent_dim,
                            saved_model_path=saved_model_path)
    # return success
    res["success"] = True

    return jsonify(res)


@app.route("/train", methods = ['GET', 'POST'])
def train_model():
    global model

    # init response
    res = {"success": False}

    # retrieve param
    videos_path = config.videos_path
    nResizeMinDim = config.nResizeMinDim

    # train model
    model.train(videos_path, nResizeMinDim)

    # return success
    res["success"] = True

    return jsonify(res)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    return None




if __name__ == "__main__":
    app.run(debug=True)
    