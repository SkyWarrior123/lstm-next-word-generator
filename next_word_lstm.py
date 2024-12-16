import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Read the dataset
with open('sherlock-holm.es_stories_plain-text_advs.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenizer process
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

# Assign length of the word text
total_words = len(tokenizer.word_index) + 1

# N-gram input sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Find the maximum sequence length
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Determining the X and y values
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Convert y to one-hot encoding
y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))

# Create the model
model = Sequential()
model.add(Embedding(total_words, 100))
model.add(LSTM(200, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# Build the model by specifying the input shape
model.build(input_shape=(None, max_sequence_len-1))

# Print the model summary
print(model.summary())


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=50, batch_size=256, verbose=1)

# Generate text using the trained model
def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Example seed texts
seed_texts = [
    "I will close the door if",
    "The case was solved by",
    "Sherlock Holmes carefully observed",
    "The mysterious figure appeared at"
]

next_words = 5

# Test the model with example seed texts
for seed_text in seed_texts:
    generated_text = generate_text(seed_text, next_words, max_sequence_len)
    print(f"Seed text: '{seed_text}'\nGenerated text: '{generated_text}'\n")
