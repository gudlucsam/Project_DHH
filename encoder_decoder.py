import keras
import numpy as np

from keras.layers import Input, Dense, LSTM
from keras.models import Model, load_model



class lstm_models():
  """
  Builds an encoder decoder (LSTM) to predict sequence of text
          
  Keyword arguments:
  nFramesTarget -- (int) number of frames in sequence
  nFeatureLength -- (int) length of features extracted from CNN per frame ( 1024 or 2048)
  max_sentence_len -- (int) character length of longest target text
  unique_char_tokens -- (int) length of unique character tokens in target text
                        used for one-hot encoding

  returns keras Model
  """

  def __init__(self, index_to_chars, chars_to_index, 
               nFramesTarget, nFeatureLength, max_sentence_len,
               unique_char_tokens, latent_dim=256, saved_model_path="saved_model/dnn.5"):
    
    self.saved_model_path = saved_model_path
    self.nFramesTarget = nFramesTarget
    self.nFeatureLength = nFeatureLength
    self.max_sentence_len = max_sentence_len
    self.num_decoder_tokens = unique_char_tokens

    # retrieve character_indexes to decode back to text
    self.index_to_chars, self.chars_to_index = index_to_chars, chars_to_index

    # latent dimensionality of the encoding space
    self.latent_dim = latent_dim

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
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
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
    # construct model for prediction
    encoder_model, decoder_model = self.construct_prediction_model()

    # ==========decode sentence back to text============

    # encode the input frames feature sequence to get the internal state vectors.
    states_value = encoder_model.predict(frames_features_sequence)
      
    # generate empty target sequence of length 1 with only the start character
    target_seq = np.zeros((1, 1, self.num_decoder_tokens))
    target_seq[0, 0, self.chars_to_index['\t']] = 1
      
    # output sequence loop
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
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

