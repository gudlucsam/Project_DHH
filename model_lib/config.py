import os


# get current dir
base_dir = os.path.dirname(os.path.abspath(__file__))

# inceptionV3Params model parameters
inception_model_params = {
    "mName" : "inception",
    "input_shape": (299, 299, 3),
    "output_shape": (2048, ),
    "saved_model_path": '\\'.join([base_dir, "saved_models\\inception\\dhh.h5"])
}

# mobilenet model parameters
mobilenet_model_params = {
    "mName" : "mobilenet",
    "input_shape": (224, 224, 3),
    "output_shape": (1024,),
    "saved_model_path": '\\'.join([base_dir, "saved_models\\mobilenet\\dhh.h5"])
}

# default model to load
params = {
    "labels_path" : '\\'.join([base_dir, "downloaded.csv"]),
    "cnn_model_params" : mobilenet_model_params,
    "nTargetFrames" : 40,
    "nFeatureLength" : mobilenet_model_params["output_shape"][0],
    "latent_dim" : 256,
    "saved_model_path" : mobilenet_model_params["saved_model_path"]
}

# videos_path = "dataset"
videos_path = '\\'.join([base_dir, "downloaded"])
nResizeMinDim = 300