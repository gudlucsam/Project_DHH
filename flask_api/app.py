import os
import numpy as np

from model import config

from model.utility_functions import target_text_encoder, process_videos
from model.encoder_decoder import lstm_models
from model.videocapture import predict_from_camera

from flask import Flask, jsonify



# instantiate flask
app = Flask(__name__)
app.debug = True

model = None



@app.route("/")
def index():
    return "Hello Word"

@app.route("/", methods=['GET'])
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
    saved_model_path= model_params["saved_model_path"]

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
    app.run()