import numpy as np

from utility_functions import token_to_index
from encoder_decoder import lstm_models


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

    # return the number of unique character token, word to index and index to words in dataset
    max_sentence_len, \
    num_uChars, \
    index_to_chars, \
    chars_to_index = token_to_index(labels_path)
    nFeatureLength = model_params["output_shape"][0]
    nTargetFrames = 40
    latent_dim = 256
    saved_model_path="saved_model/dnn.5"

    # build encoder - decoder model
    instance = lstm_models(index_to_chars, chars_to_index,
                            nTargetFrames, nFeatureLength,
                            max_sentence_len, num_uChars)
    # train model
    model = instance.encoder_decoder_model()

   