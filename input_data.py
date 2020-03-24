def target_text_encoder(labels_path):
    # Vectorize the data.
    target_texts = []
    target_characters = set()
    # read target text data
    df = pd.read_csv(labels_path)
    target_samples = df['translation'].values.tolist()
    for target_text in target_samples:
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        target_texts.append(target_text)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    target_characters = sorted(list(target_characters))
    num_decoder_tokens = len(target_characters)
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(target_texts))
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

    decoder_input_data = np.zeros(
        (len(target_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(target_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for i, target_text in enumerate(target_texts):
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1
        decoder_input_data[i, t + 1:, target_token_index[' ']] = 1
        decoder_target_data[i, t:, target_token_index[' ']] = 1

    return max_decoder_seq_length, num_decoder_tokens, decoder_input_data, decoder_target_data