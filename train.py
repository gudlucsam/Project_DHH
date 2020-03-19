import numpy as np

from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau

from utility_functions import token_to_index
from feature_extraction import features_generator
from cnn_model import features_2D_model
from encoder_decoder import lstm_models


if __name__ == "__main__":

    
    videos_path = "dataset"
    labels_path = "dataset_labels.csv"

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

    # return the number of unique character token, word to index and index to words in dataset
    max_sentence_len, num_uChars, index_to_chars, chars_to_index = token_to_index(labels_path)

    # build CNN for feature extraction
    feature_extraction_model = features_2D_model(**model_params)
    
    # extract features using CNN, and process frames
    encoder_input_data, decoder_input_data = features_generator(videos_path, labels_path, feature_extraction_model,
                                                        max_sentence_len=max_sentence_len, num_chars=num_uChars,
                                                        nTargetFrames=40, nResizeMinDim=256)

    # initialize target data without start characters
    decoder_target_data = np.zeros(decoder_input_data.shape, dtype="int32")
    decoder_target_data[:, 0:-1, :] = decoder_input_data[:, 1:, :]

    # initialize target data
    # test_decoder_target_data = np.zeros(test_decoder_input_data.shape, dtype="int32")
    # test_decoder_target_data[:, 0:-1, :] = test_decoder_input_data[:, 1:, :]


    # build encoder - decoder model
    print("Building encoder - decoder model for training...")
    nFeatureLength = model_params["output_shape"][0]
    instance = lstm_models(index_to_chars, chars_to_index, 40, nFeatureLength, max_sentence_len, num_uChars)
    model = instance.encoder_decoder_model()
    print("Done")

    # callbacks
    early_stopping = EarlyStopping(monitor='val_loss', 
                                    min_delta=0, patience=5, 
                                    verbose=0, mode='auto', 
                                    baseline=None, 
                                    restore_best_weights=False)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                    patience=3, min_lr=0.001)

    # compile model to train
    print("Compiling model....")
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # train lstm model
    print("Training model....")
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                batch_size=5, epochs=50, validation_split=0.2)

    # save model
    print("Saving model....")
    model.save('/saved_model/dnn.h5')

    print("Done")