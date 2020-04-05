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
# keras model
model = None


@app.route("/ml/api/v1.0/info")
def index():
    return jsonify({
        "project_name":"Undergraduate Capstone Project",
        "model_name":"dhh-asl-api-v1.0",
        "version":"v1.0",
        "author":"Samuel Atule",
        "email":"atulesamuel20@gmail.com"
    })

@app.route("/ml/api/v1.0/md/<int:model_id>", methods=['GET'])
def select_feature_extraction_model(model_id):

    # init response
    res = {"success": False}

    # make model instance global
    global model

    # select inception if model_id is 0 else mobilenet
    if model_id == 0:
        config.params["cnn_model_params"] = config.inception_model_params
        config.params["nFeatureLength"] = config.inception_model_params["output_shape"]
        config.params["saved_model_path"] = config.inception_model_params["saved_model_path"]
    elif model_id == 1:
        config.params["cnn_model_params"] = config.mobilenet_model_params
        config.params["nFeatureLength"] = config.mobilenet_model_params["output_shape"]
        config.params["saved_model_path"] = config.mobilenet_model_params["saved_model_path"]
    else:
        return jsonify(res)

    # build encoder - decoder model
    model = lstm_models(**config.params)

    # return success
    res["success"] = True

    return jsonify(res)


@app.route("/ml/api/v1.0/train", methods = ['GET', 'POST'])
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


@app.route("/ml/api/v1.0/predict", methods=['GET', 'POST'])
def predict():
    return None




if __name__ == "__main__":
    print("LOADING DEFAULT MODEL....")
    model = lstm_models(**config.params)
    app.run(debug=True)
    