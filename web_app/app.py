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



# GLOBAL VARS
app = Flask(__name__)
model = None
liFrames = []
mode = 0
# retrieve params from config
nTargetFrames = config.params["nTargetFrames"]
nHeight, nWidth, _ = config.params["cnn_model_params"]["input_shape"]



@app.route('/')
def index():
    select_form_vals = [(0, "Mobilenet(Recommended for slow device)"), (1, "InceptionV3")]
    mode = 0
    prediction = ""
    if request.args.get("mode"):
        mode = request.args.get("mode")

    if request.args.get("prediction"):
        prediction = request.args.get("prediction")

    return render_template('index.html', mode=mode, form_vals=select_form_vals, prediction=prediction)


@app.route("/model", methods=['POST'])
def select_feature_extraction_model():
    # make model instance global
    global model

    if request.method == 'POST':
        mode = 0
        model_id = int(request.form.get("selectmodel"))

        # select inception if model_id is 0 else mobilenet
        if model_id == 0:
            config.params["cnn_model_params"] = config.mobilenet_model_params
            config.params["nFeatureLength"] = config.mobilenet_model_params["output_shape"]
            config.params["saved_model_path"] = config.mobilenet_model_params["saved_model_path"]
            #  select chosen option
            mode = 0

        elif model_id == 1:
            config.params["cnn_model_params"] = config.inception_model_params
            config.params["nFeatureLength"] = config.inception_model_params["output_shape"]
            config.params["saved_model_path"] = config.inception_model_params["saved_model_path"]
            #  select chosen option
            mode = 1

        else:
            mode = 0
            return redirect( url_for('index', mode=mode))

        # build encoder - decoder model
        model = lstm_models(**config.params)

        # retrieve param
        videos_path = config.videos_path
        nResizeMinDim = config.nResizeMinDim

        # train or load model
        model.train(videos_path, nResizeMinDim)

        return redirect( url_for('index', mode=mode) )


@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        global model
        global liFrames
        # retrieve params from config
        global nTargetFrames
        global nHeight, nWidth

        # process frames
        liFrames = np.array(liFrames)
        video_frames = images_normalize(liFrames, nTargetFrames, nHeight, nWidth)

        # predict from live feeds
        prediction = model.predict([video_frames])

        # reset list
        liFrames = []
        return redirect( url_for('index', prediction=prediction) )
    else:
        return redirect( url_for('index') )


def gen(camera):
    global liFrames
    liFrames = []
    while True:
        # capsture frames
        frame, img = camera.get_frame()
        # append images for prediction
        liFrames.append(img)

        # yield image bytes to stream to web
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed', methods=["GET", "POST"])
def video_feed():
    global liFrames
    global mode
    if request.method == 'POST':
        liFrames = []
        return redirect(url_for('index', mode=mode))
    else:
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
    