# return the number of unique character token, word to index and index to words 
# in the training dataset
trainLabelsPath = "/gdrive/My Drive/Capstone/labels/train_labels.csv"
max_sentence_len, num_uChars, index_to_chars, chars_to_index = token_to_index(trainLabelsPath)


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

# build CNN feature extractor
feature_extractor = features_2D_model(**model_params)

# ======================Training Data===========================
# extract features from video frames and save to .npy files
pickleDirPath = "/gdrive/My Drive/Capstone/frames_npy/train.npy"
labelsPath = "/gdrive/My Drive/Capstone/labels/train_labels.csv"
sFeatureBaseDir = "/gdrive/My Drive/Capstone/2d_features/train/"
sVideosPath = "/gdrive/My Drive/Capstone/dataset/train/"

# extract features using CNN and save to disc
features_2D_generator(sVideosPath, pickleDirPath, labelsPath, sFeatureBaseDir, feature_extractor, nFramesNorm=40)

# load extracted features from disc to array to train LSTM
print("loading saved features for training...")
train_encoder_input_data, train_decoder_input_data = npyLoad(sFeatureBaseDir)
print("Done")

# initialize target data
train_decoder_target_data = np.zeros(train_decoder_input_data.shape, dtype="int32")
train_decoder_target_data[:, 0:-1, :] = train_decoder_input_data[:, 1:, :] #without start characters
# =======================End======================================


# ======================Validation Data===========================
# extract features from video frames and save to .npy files
pickleDirPath = "/gdrive/My Drive/Capstone/frames_npy/validation.npy"
labelsPath = "/gdrive/My Drive/Capstone/labels/validation_labels.csv"
sFeatureBaseDir = "/gdrive/My Drive/Capstone/2d_features/validation"
sVideosPath = "/gdrive/My Drive/Capstone/dataset/validation/"

# extract features using CNN and save to disc
features_2D_generator(sVideosPath, pickleDirPath, labelsPath, sFeatureBaseDir, feature_extractor, nFramesNorm=40)

# load extracted features from disc to array to validate LSTM
print("loading saved features for validation...")
validation_encoder_input_data, validation_decoder_input_data = npyLoad(sFeatureBaseDir)
print("Done")

# initialize target data
validation_decoder_target_data = np.zeros(validation_decoder_input_data.shape, dtype="int32")
validation_decoder_target_data[:, 0:-1, :] = validation_decoder_input_data[:, 1:, :] #without start characters
# =======================End======================================


# ======================Testing Data==============================
# extract features from video frames and save to .npy files
pickleDirPath = "/gdrive/My Drive/Capstone/frames_npy/test.npy"
labelsPath = "/gdrive/My Drive/Capstone/labels/test_labels.csv"
sFeatureBaseDir = "/gdrive/My Drive/Capstone/2d_features/test/"
sVideosPath = "/gdrive/My Drive/Capstone/dataset/test/"

# extract features using CNN and save to disc
features_2D_generator(sVideosPath, pickleDirPath, labelsPath, sFeatureBaseDir, feature_extractor, nFramesNorm=40)

# load extracted features from disc to array to test LSTM
print("loading saved features for testing...")
test_encoder_input_data, test_decoder_input_data = npyLoad(sFeatureBaseDir)
print("Done")

# initialize target data
test_decoder_target_data = np.zeros(test_decoder_input_data.shape, dtype="int32")
test_decoder_target_data[:, 0:-1, :] = test_decoder_input_data[:, 1:, :]
# =======================End=======================================


# ========================Building Model===========================
# train lstm model
print("Building encoder - decoder model for training...")
nFeatureLength = model_params["output_shape"][0]
instance = lstm_models(index_to_chars, chars_to_index, 40, nFeatureLength, max_sentence_len, num_uChars)
model = instance.encoder_decoder_model()
print("Done")

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0, patience=5, 
                                        verbose=0, mode='auto', 
                                        baseline=None, 
                                        restore_best_weights=False)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.001)

# compile model to train
print("Training model....")
model.compile(optimizer='rmsprop',
              #metrics=[levenshtein_cer],
              loss='categorical_crossentropy')
model.fit([train_encoder_input_data, train_decoder_input_data], train_decoder_target_data,
          batch_size=5,
          epochs=50,
          validation_data=([validation_encoder_input_data, validation_decoder_input_data],
                           validation_decoder_target_data),
          callbacks=[early_stopping]
          )

# =======================Saving Model============================
model.save('/gdrive/My Drive/Capstone/dnn.h5')

print("Done")