import keras
import numpy as np

from utility_functions import token_to_index
from feature_extraction import features_generator
from cnn_model import features_2D_model
from encoder_decoder import lstm_models

from keras.models import load_model, Model
from keras.layers import Input, Dense, LSTM
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau


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


class p_model():
  def __init__(self, saved_model_path, latent_dim=256):
    self.latent_dim = latent_dim
    self.saved_model_path = saved_model_path

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

if __name__ == "__main__":
  saved_model_path = "saved_model/dnn.h5"
  instance = p_model(saved_model_path, latent_dim=256)
  print(instance.construct_prediction_model())
  # prediction of frames
  # for frames_sequence in train.encoder_input_data:
  #   print(frames_sequence)
    # frames_sequence = np.expand_dims(frames_sequence, axis=0) 
    # decoded_sentence = instance.decode_frame_sequence(frames_sequence)
    # print(decoded_sentence)