# inceptionV3Params model parameters
inception_model_params = {
    "mName" : "inception",
    "input_shape": (299, 299, 3),
    "output_shape": (2048, ),
    "saved_model_path": "saved_models/inception/dhh.h5"
}
# mobilenet model parameters
mobilenet_model_params = {
    "mName" : "mobilenet",
    "input_shape": (224, 224, 3),
    "output_shape": (1024,),
    "saved_model_path": "saved_models/mobilenet/dhh.h5"
}

# videos_path = "dataset"
videos_path = "downloaded"
labels_path = "downloaded.csv"

nTargetFrames = 40
latent_dim = 256
nResizeMinDim = 300