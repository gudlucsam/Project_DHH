import os
import time
import sys

import numpy as np

sys.path.append('C:\\Users\\atule\\Desktop\\Project_DNN')

from model_lib import config
from model_lib.utility_functions import target_text_encoder, process_videos, images_normalize
from model_lib.encoder_decoder import lstm_models
from model_lib.videocapture import predict_from_camera
from model_lib.camera import VideoCamera

from flask_cors import CORS
from flask import Flask, jsonify, make_response, Response, request, render_template, redirect, url_for




# instantiate flask
app = Flask(__name__)
# keras model
model = None
# frames list
liFrames = []
status = True

@app.route('/')
def index():
    select_form_vals = [(0, "Mobilenet(Recommended for slow device)"), (1, "InceptionV3")]
    default = 0
    if request.args.get("default_val"):
        default = request.args.get("default_val")
    return render_template('index.html', default_val=default, form_vals=select_form_vals)


@app.route("/model", methods=['POST'])
def select_feature_extraction_model():

    # make model instance global
    global model

    if request.method == 'POST':
        default = 0
        model_id = int(request.form.get("selectmodel"))

        # select inception if model_id is 0 else mobilenet
        if model_id == 0:
            config.params["cnn_model_params"] = config.mobilenet_model_params
            config.params["nFeatureLength"] = config.mobilenet_model_params["output_shape"]
            config.params["saved_model_path"] = config.mobilenet_model_params["saved_model_path"]
            #  select chosen option
            default = 0

        elif model_id == 1:
            config.params["cnn_model_params"] = config.inception_model_params
            config.params["nFeatureLength"] = config.inception_model_params["output_shape"]
            config.params["saved_model_path"] = config.inception_model_params["saved_model_path"]
            #  select chosen option
            default = 1

        else:
            default = 0
            return redirect( url_for('index', default_val=default))

        # build encoder - decoder model
        model = lstm_models(**config.params)

        # retrieve param
        videos_path = config.videos_path
        nResizeMinDim = config.nResizeMinDim

        # train or load model
        model.train(videos_path, nResizeMinDim)

        return redirect( url_for('index', default_val=default))


def predict():
    global model
    global liFrames

    # retrieve params from config
    nTargetFrames = config.params["nTargetFrames"]
    nHeight, nWidth, _ = config.params["cnn_model_params"]["input_shape"]

    # process frames
    liFrames = np.array(liFrames)
    video_frames = images_normalize(liFrames, nTargetFrames, nHeight, nWidth)

    # predict from live feeds
    prediction = model.predict([video_frames])

    return prediction


def gen(camera, nTimeDuration = 4):
    global liFrames
    # fTimeStart = time.time()

    liFrames = []
    while True:
        # stop after nTimeDuration sec
        # fTimeElapsed = time.time() - fTimeStart
        # if fTimeElapsed > nTimeDuration: break
        # capsture frames
        frame, img = camera.get_frame()
        # append images for prediction
        liFrames.append(img)

        # yield image bytes to stream to web
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/ml/api/v1.0/md/vf')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == "__main__":
    print("LOADING DEFAULT MODEL....")
    model = lstm_models(**config.params)
    # retrieve param
    videos_path = config.videos_path
    nResizeMinDim = config.nResizeMinDim
    # train or load model
    model.train(videos_path, nResizeMinDim)

    # start app engine
    app.run(debug=True, threaded = True)
    