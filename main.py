import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

def load_data_and_model():
    dataset = pd.read_csv('/home/yas/clg_codes/lyrics_generator/lyrics_generator/data.csv')
    dataset = dataset.head(500)
    dataset['Word_Count'] = dataset['lyrics'].apply(lambda x: len(str(x).split()))

    text_data = dataset['lyrics'].astype(str).str.lower()
    token_maker = Tokenizer()
    token_maker.fit_on_texts(text_data)
    sequences = token_maker.texts_to_sequences(text_data)

    sequences_padded = []
    for seq in sequences:
        for i in range(1, len(seq)):
            n_gram_sequence = seq[:i+1]
            sequences_padded.append(n_gram_sequence)

    max_sequence_length = max([len(seq) for seq in sequences_padded])
    sequences_padded = np.array(pad_sequences(sequences_padded, maxlen=max_sequence_length, padding='pre'))
    
    model = load_model('/home/yas/clg_codes/lyrics_generator/lyrics_generator/song_lyrics_generator.h5')
    return token_maker, max_sequence_length, model

def complete_this_song(model, token_maker, max_sequence_length, starting_text, num_words):
    for _ in range(num_words):
        token_list = token_maker.texts_to_sequences([starting_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        probabilities = model.predict(token_list)[0]
        predicted_index = np.argmax(probabilities)

        output_word = ""
        for word, index in token_maker.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        starting_text += " " + output_word
    return starting_text

# token_maker, max_sequence_length, model = load_data_and_model()
# input_text = 'rest your head on my shoulder'
# generated_lyrics = generate_lyrics(model, token_maker, max_sequence_length, input_text, 50)
# print(generated_lyrics)
