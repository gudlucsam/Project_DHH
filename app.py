import os
import time
import numpy as np

from model_lib import config

from model_lib.utility_functions import target_text_encoder, process_videos, images_normalize
from model_lib.encoder_decoder import lstm_models
from model_lib.videocapture import predict_from_camera
from model_lib.camera import VideoCamera

from flask import Flask, jsonify, make_response, Response

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

# instantiate flask
app = Flask(__name__)
# keras model
model = None
graph = None
liFrames = []

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


@app.route("/ml/api/v1.0/md/tn", methods = ['GET', 'POST'])
def train_model():
    global model
    global graph

    # init response
    res = {"success": False}

    # retrieve param
    videos_path = config.videos_path
    nResizeMinDim = config.nResizeMinDim

    # train model
    model.train(videos_path, nResizeMinDim)
    graph = model.graph

    # return success
    res["success"] = True

    return jsonify(res)

def gen(camera, nTimeDuration = 4):
    global liFrames
    fTimeStart = time.time()

    liFrames = []
    while True:
        # stop after nTimeDuration sec
        fTimeElapsed = time.time() - fTimeStart
        if fTimeElapsed > nTimeDuration: break
        # capsture frames
        frame, img = camera.get_frame()
        # append images for prediction
        liFrames.append(img)

        # yield image bytes to stream to web
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/ml/api/v1.0/md/vf')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/ml/api/v1.0/md/pd")
def predict():
    global model
    global liFrames
    global graph

    # init response
    res = {
        "success": False,
        "prediction": []
        }

    # retrieve params from config
    nTargetFrames = config.params["nTargetFrames"]
    nHeight, nWidth, _ = config.params["cnn_model_params"]["input_shape"]

    # process frames
    liFrames = np.array(liFrames)
    video_frames = images_normalize(liFrames, nTargetFrames, nHeight, nWidth)
    print("xxxxxxxxxxxxxxxxxxx", video_frames.shape)
    print("....rrrrrrrrrrrrrrrrr", video_frames)

    # predict from live feeds
    with graph.as_default():
        prediction = model.predict([video_frames])
    
    # check if prediction successful otherwise return unsuccessful
    if not prediction:
        return jsonify(res)

    # return success
    res["success"] = True
    res["prediction"] = prediction

    return jsonify(res)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == "__main__":
    print("LOADING DEFAULT MODEL....")
    model = lstm_models(**config.params)
    app.run(debug=True)
    