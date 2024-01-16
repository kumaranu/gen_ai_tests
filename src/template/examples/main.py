import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Replace this with your own text data or use a dataset of your choice
text_data = "It was the best of times, it was the worst of times," \
            " it was the age of wisdom, it was the age of foolishness," \
            " it was the epoch of belief, it was the epoch of incredulity," \
            " it was the season of Light, it was the season of Darkness," \
            " it was the spring of hope, it was the winter of despair," \
            " we had everything before us, we had nothing before us," \
            " we were all going direct to Heaven," \
            " we were all going direct the other way – in short," \
            " the period was so far like the present period," \
            " that some of its noisiest authorities insisted on its being received," \
            " for good or for evil, in the superlative degree of comparison only."

# Create a mapping of unique characters to indices
chars = sorted(list(set(text_data)))
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}

# Create input sequences and corresponding labels
seq_length = 100
sequences = []
next_chars = []

for i in range(0, len(text_data) - seq_length, 1):
    seq = text_data[i:i + seq_length]
    next_char = text_data[i + seq_length]
    sequences.append([char_to_index[char] for char in seq])
    next_chars.append(char_to_index[next_char])

# Reshape data for LSTM
X = tf.reshape(sequences, (len(sequences), seq_length, 1))
X = tf.cast(X, tf.float32) / float(len(chars))
X = X / float(len(chars))  # Normalize input data

y = tf.keras.utils.to_categorical(next_chars)

# Build the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(256))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, epochs=500, batch_size=64)


# Function to generate text
def generate_text(seed_text, next_words, model, max_sequence_len, temperature=1.0):
    generated_text = seed_text
    for _ in range(next_words):
        token_list = [char_to_index.get(char, 0) for char in seed_text]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted = model.predict(token_list, verbose=0)

        # Adjust temperature to control randomness in predictions
        predicted = np.log(predicted) / temperature
        exp_preds = np.exp(predicted)
        predicted_probs = exp_preds / np.sum(exp_preds)

        # Sample from the predicted probabilities
        predicted_index = np.random.choice(len(predicted_probs[0]), p=predicted_probs[0])

        output_char = index_to_char.get(predicted_index, '')
        seed_text += output_char
        generated_text += output_char
    return generated_text


# Generate text using the trained model
# generated_text = generate_text("Your seed text goes here.", 100, model, seq_length)
generated_text = generate_text("It was the", 100, model, seq_length)

print(generated_text)
