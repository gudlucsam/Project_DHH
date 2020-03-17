 # prediction of frames
for frames_sequence in validation_decoder_input_data:
  frames_sequence = np.expand_dims(frames_sequence, axis=0) 
  decoded_sentence = instance.decode_frame_sequence(frames_sequence)
  print(decoded_sentence)