import tensorflow as tf


class RNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units, batch_size):
        super(RNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None])
        self.rnn = tf.keras.layers.SimpleRNN(hidden_units, return_sequences=False)
        self.output_layer = tf.keras.layers.Dense(2, activation=None)

    def call(self, x):
        embedding_layer = self.embedding(x)
        print(embedding_layer.shape)
        features = self.rnn(embedding_layer)
        print(features.shape)
        logits = self.output_layer(features)
        print(logits.shape)
        return logits