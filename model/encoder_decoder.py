import os
import keras
import numpy as np

from feature_extraction import features_generator
from cnn_model import features_2D_model
from utility_functions import target_text_encoder

from keras.layers import Input, Dense, LSTM
from keras.models import Model, load_model
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau


class lstm_models():
  """
  Builds an encoder decoder (LSTM) to predict sequence of text
          
  Keyword arguments:
  nTargetFrames -- (int) number of frames in sequence
  nFeatureLength -- (int) length of features extracted from CNN per frame ( 1024 or 2048)
  max_sentence_len -- (int) character length of longest target text
  unique_char_tokens -- (int) length of unique character tokens in target text
                        used for one-hot encoding

  returns keras Model
  """

  def __init__(self, labels_path, cnn_model_params, nTargetFrames, nFeatureLength, 
              latent_dim=256, saved_model_path="saved_models/mobilenet/dnn.h5"):
    
    
    # dataset stats
    self.nTargetFrames = nTargetFrames
    self.nFeatureLength = nFeatureLength
    # latent dimensionality of the encoding space
    self.latent_dim = latent_dim
    # dir to save trained model
    self.saved_model_path = saved_model_path
    # return the number of unique character token, word to index and index to words in dataset
    self.max_sentence_len, \
    self.num_decoder_tokens, \
    self.chars_to_index, \
    self.index_to_chars, \
    self.decoder_input_data, \
    self.decoder_target_data = target_text_encoder(labels_path)

    # build CNN for feature extraction
    self.feature_extraction_model = features_2D_model(
            cnn_model_params["mName"],
            cnn_model_params["input_shape"],
            cnn_model_params["output_shape"])


  def encoder_decoder_model(self):
    # inputs to encoder, decoder
    encoder_inputs = Input(shape=(None, self.nFeatureLength))
    decoder_inputs = Input(shape=(None, self.num_decoder_tokens))

    # encoder, decoder and dense models
    encoder = LSTM(self.latent_dim, recurrent_dropout=0.25, dropout=0.25, return_state=True)
    decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
    decoder_lstm = LSTM(self.latent_dim, recurrent_dropout=0.25, dropout=0.25, return_sequences=True, 
                              return_state=True)

    # retrieve encoder states to use as initial states of decoder model
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # retrieve outputs of decoder
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    # feed decoder outputs to dense layer for final prediction
    outputs = decoder_dense(decoder_outputs)
    # construct encoder-decoder model using Keras functional API
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

    return model


  def construct_prediction_model(self):
    model = load_model(self.saved_model_path)

    encoder_inputs = model.input[0]   # input_1
    _, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]   # input_2
    decoder_state_input_h = Input(shape=(self.latent_dim,), name='input_4')
    decoder_state_input_c = Input(shape=(self.latent_dim,), name='input_5')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]

    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model

  def decode_frame_sequence(self, frames_features_sequence):

    # convert (40, 1024) to (1, 40, 1024)
    frames_features_sequence = np.expand_dims(frames_features_sequence, axis=0)
    
    # encode the input frames feature sequence to get the internal state vectors.
    states_value = self.encoder_model.predict(frames_features_sequence)
      
    # generate empty target sequence of length 1 with only the start character
    target_seq = np.zeros((1, 1, self.num_decoder_tokens))
    target_seq[0, 0, self.chars_to_index['\t']] = 1
      
    # output sequence loop
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
        
      # sample a token and add the corresponding character to the decoded sequence
      sampled_token_index = np.argmax(output_tokens[0, -1, :])
      sampled_char = self.index_to_chars[sampled_token_index]
      decoded_sentence += sampled_char
        
      # check for the exit condition: either hitting max length or predicting the 'stop' character
      if (sampled_char == '\n' or len(decoded_sentence) > self.max_sentence_len):
        stop_condition = True
          
      # update the target sequence (length 1).
      target_seq = np.zeros((1, 1, self.num_decoder_tokens))
      target_seq[0, 0, sampled_token_index] = 1.
        
      # update states
      states_value = [h, c]
        
    return decoded_sentence 

  def train(self, videos_path, nResizeMinDim):
    if os.path.exists(self.saved_model_path):
      print("Model already trained and saved to", self.saved_model_path)
      print("Training stopping...")

      # construct model for prediction
      print("reconstructing models for prediction....")
      self.encoder_model, self.decoder_model = self.construct_prediction_model()
    
    else:
       
      # extract features using CNN, and process frames
      encoder_input_data = features_generator(videos_path, self.feature_extraction_model,
                                  nTargetFrames=self.nTargetFrames, nResizeMinDim=nResizeMinDim)

      # construct encoder -decoder model for training
      print("Building encoder - decoder model for training...")
      model = self.encoder_decoder_model()

      # callbacks
      early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                      verbose=0, mode='auto', baseline=None, restore_best_weights=False)
      reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

      # compile model to train
      print("Compiling model....")
      model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

      # train lstm model
      print("Training model....")
      model.fit([encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                  batch_size=5, epochs=20, validation_split=0.2)

      # create dir if not exists
      print("Saving model....")
      if not os.path.exists(self.saved_model_path):
        os.makedirs("/".join(self.saved_model_path.split("/")[:-1]), exist_ok=True)

      # save model
      model.save(self.saved_model_path)

      print("Done")

      # construct model for prediction
      print("reconstructing models for prediction....")
      self.encoder_model, self.decoder_model = self.construct_prediction_model()

  def predict(self, frames_sequence):
    """
    predict using sequences of frames
    """
    
    sentences = []
    for sequence in frames_sequence:
      # extract features from cnn model
      feature_frames = self.feature_extraction_model.predict(sequence)
      # predict using features
      predicted_sentence= self.decode_frame_sequence(feature_frames)
      sentences.append(predicted_sentence)
      # print(predicted_sentence)

    return sentences
