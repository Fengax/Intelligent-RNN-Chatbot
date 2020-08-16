import tensorflow as tf
import os
import numpy as np

text = open("text.txt", "rb").read().decode(encoding="utf-8")

vocab = sorted(set(text))
char_to_index = {}
index_to_char = np.array(vocab)

for i, u in enumerate(vocab):
    char_to_index[u] = i

def encode_text(text):
    return np.array([char_to_index[c] for c in text])

def decode_text(int):
    return "".join(index_to_char[int])

def split_input(chunk):
    inp = chunk[:-1]
    target = chunk[1:]
    return inp, target

num_examples = len(text) // 101
char_dataset = tf.data.Dataset.from_tensor_slices(encode_text(text))

sequences = char_dataset.batch(101, drop_remainder=True)
dataset = sequences.map(split_input)

batch_size = 64
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
buffer_size = 10000

data = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

checkpoint_dir = './training_checkpoints'


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
    tf.keras.layers.Dense(vocab_size)
])


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

if os.path.isdir("./training_checkpoints"):
    inp = input("Checkpoint detected. Continue training? (Y/N) Enter N if you want to use the current trained checkpoint")
    if inp == "Y":
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            monitor="loss",
            filepath=checkpoint_prefix,
            save_weights_only=True,
            save_best_only=True,
            mode='min')

        history = model.fit(data, epochs=500, callbacks=[checkpoint_callback])
    else:
        inp = input("Use current trained checkpoint? (Y/N) Enter N if you want to retrain the entire model.")
        if inp == "Y":
            pass
        else:
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                monitor="loss",
                filepath=checkpoint_prefix,
                save_weights_only=True,
                save_best_only=True,
                mode='min')

            history = model.fit(data, epochs=500, callbacks=[checkpoint_callback])
else:
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        monitor="loss",
        filepath=checkpoint_prefix,
        save_weights_only=True,
        save_best_only=True,
        mode='min')

    history = model.fit(data, epochs=500, callbacks=[checkpoint_callback])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[1, None]),
    tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
    tf.keras.layers.Dense(vocab_size)
])

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
    num_generate = 800

    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(index_to_char[predicted_id])

    return (start_string + ''.join(text_generated))


while True:
    inp = input()
    print(generate_text(model, inp))
