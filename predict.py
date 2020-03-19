import train
import numpy as np

from keras.models import load_model
import train


if __name__ == "__main__":

  # prediction of frames
  for frames_sequence in train.encoder_input_data:
    print(frames_sequence)
    # frames_sequence = np.expand_dims(frames_sequence, axis=0) 
    # decoded_sentence = instance.decode_frame_sequence(frames_sequence)
    # print(decoded_sentence)