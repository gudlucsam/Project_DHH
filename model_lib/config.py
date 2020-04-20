# inceptionV3Params model parameters
inception_model_params = {
    "mName" : "inception",
    "input_shape": (299, 299, 3),
    "output_shape": (2048, ),
    "saved_model_path": "C:\\Users\\atule\\Desktop\\Project_DNN\\model_lib\\saved_models\\inception\\dhh.h5"
}

# mobilenet model parameters
mobilenet_model_params = {
    "mName" : "mobilenet",
    "input_shape": (224, 224, 3),
    "output_shape": (1024,),
    "saved_model_path": "C:\\Users\\atule\\Desktop\\Project_DNN\\model_lib\\saved_models\\mobilenet\\dhh.h5"
}

# default model to load
params = {
    "labels_path" : "C:\\Users\\atule\\Desktop\\Project_DNN\\model_lib\\downloaded.csv",
    "cnn_model_params" : mobilenet_model_params,
    "nTargetFrames" : 40,
    "nFeatureLength" : mobilenet_model_params["output_shape"][0],
    "latent_dim" : 256,
    "saved_model_path" : mobilenet_model_params["saved_model_path"]
}

# videos_path = "dataset"
videos_path = "C:\\Users\\atule\\Desktop\\Project_DNN\\model_lib\\downloaded"
nResizeMinDim = 300