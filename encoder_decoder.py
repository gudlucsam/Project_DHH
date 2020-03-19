import keras
import numpy as np

from keras.layers import Input, Dense, LSTM
from keras.models import Model



class lstm_models():
  """Builds an encoder decoder (LSTM) to predict sequence of text
          
        Keyword arguments:
        nFramesTarget -- (int) number of frames in sequence
        nFeatureLength -- (int) length of features extracted from CNN per frame ( 1024 or 2048)
        max_sentence_len -- (int) character length of longest target text
        unique_char_tokens -- (int) length of unique character tokens in target text
                              used for one-hot encoding

        returns keras Model
        """

  def __init__(self, index_to_chars, chars_to_index, 
               nFramesTarget, nFeatureLength, max_sentence_len, unique_char_tokens, latent_dim=256):
    
    self.nFramesTarget = nFramesTarget
    self.nFeatureLength = nFeatureLength
    self.max_sentence_len = max_sentence_len
    self.num_decoder_tokens = unique_char_tokens

    # retrieve character_indexes to decode back to text
    self.index_to_chars, self.chars_to_index = index_to_chars, chars_to_index

    # latent dimensionality of the encoding space
    self.latent_dim = latent_dim

    # inputs to encoder, decoder
    self.encoder_inputs = Input(shape=(None, self.nFeatureLength))
    self.decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
    

    # construct encoder, decoder and dense models for use in training and prediction models
    self.encoder = LSTM(self.latent_dim, recurrent_dropout=0.25, dropout=0.25, return_state=True)


    self.decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
    
    self.decoder_lstm = LSTM(self.latent_dim, recurrent_dropout=0.25,
                              dropout=0.25, return_sequences=True, return_state=True)


  def encoder_decoder_model(self):
      # retrieve encoder states to use as initial states of decoder model
      encoder_outputs, state_h, state_c = self.encoder(self.encoder_inputs)
      self.encoder_states = [state_h, state_c]

      # retrieve outputs of decoder
      decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs, initial_state=self.encoder_states)

      # feed encoder outputs to dense layer for final prediction
      decoder_outputs = self.decoder_dense(decoder_outputs)

      # construct encoder-decoder model using Keras functional API
      model = Model(inputs=[self.encoder_inputs, self.decoder_inputs], outputs=decoder_outputs)

      return model

  def construct_prediction_model(self):
      # separate encoder model to encode input feature frames sequences 
      pEncoderModel = Model(self.encoder_inputs, self.encoder_states)

      # specify decoder states input shape
      decoder_state_input_h = Input(shape=(self.latent_dim,))
      decoder_state_input_c = Input(shape=(self.latent_dim,))
      decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
      #  construct separate decoder for prediction, and retrieve decoder states
      decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_inputs,
                                                              initial_state=decoder_states_inputs)
      decoder_states = [state_h, state_c]

      # predict characters with dense layers by using decoder outputs as inputs
      decoder_outputs = self.decoder_dense(decoder_outputs)

      # decoder model for prediction of characters
      dDecoderModel = Model([self.decoder_inputs] + decoder_states_inputs,
                                              [decoder_outputs] + decoder_states)
        
      return pEncoderModel, dDecoderModel
         

  def decode_frame_sequence(self, frames_features_sequence):
      # construct model for prediction
      pEncoderModel, dDecoderModel = self.construct_prediction_model()

      # ==========decode sentence back to text============

      # encode the input frames feature sequence to get the internal state vectors.
      states_value = pEncoderModel.predict(frames_features_sequence)
        
      # generate empty target sequence of length 1 with only the start character
      target_seq = np.zeros((1, 1, self.num_decoder_tokens))
      target_seq[0, 0, self.chars_to_index['\t']] = 1
        
      # output sequence loop
      stop_condition = False
      decoded_sentence = ''
      while not stop_condition:
        output_tokens, h, c = dDecoderModel.predict(
          [target_seq] + states_value)
          
        # sample a token and add the corresponding character to the 
        # decoded sequence
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = self.index_to_chars[sampled_token_index]
        decoded_sentence += sampled_char
          
        # check for the exit condition: either hitting max length
        # or predicting the 'stop' character
        if (sampled_char == '\n' or len(decoded_sentence) > self.max_sentence_len):
          stop_condition = True
            
        # update the target sequence (length 1).
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
          
        # update states
        states_value = [h, c]
          
      return decoded_sentence 

