import tensorflow as tf

from chat_box.bahdanau_attention import BahdanauAttention

class Decoder(tf.keras.Model):
    """
    Decoder module for sequence-to-sequence models.

    This class implements the decoder module for sequence-to-sequence models. It takes an input token,
    a hidden state, and the encoder output as inputs and produces the output token, the new hidden state,
    and the attention weights.

    Args:
    - vocab_size (int): The size of the vocabulary.
    - embedding_dim (int): The dimensionality of the embedding layer.
    - dec_units (int): The number of units in the decoder GRU layer.
    - batch_sz (int): The batch size.

    Methods:
    - call(y, hidden, enc_output): Performs the forward pass of the decoder.
      - y: The input token.
      - hidden: The hidden state.
      - enc_output: The encoder output.
      Returns the output token, the new hidden state, and the attention weights.

    Note:
    - This implementation uses the Bahdanau attention mechanism.
    """
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, y, hidden, enc_output):
        """
        Performs the forward pass of the decoder.

        Args:
        - y: The input token.
        - hidden: The hidden state.
        - enc_output: The encoder output.

        Returns:
        - y: The output token.
        - state: The new hidden state.
        - attention_weights: The attention weights.

        The decoder takes an input token, a hidden state, and the encoder output as inputs.
        It first computes the attention weights and the context vector using the Bahdanau attention mechanism.
        Then, it embeds the input token and concatenates it with the context vector.
        The concatenated vector is fed into the GRU layer, which produces an output and a new hidden state.
        Finally, the output is reshaped and passed through a dense layer to obtain the output token.

        Note:
        - The input token, hidden state, and encoder output should have compatible shapes for matrix multiplication.
        """
        context_vector, attention_weights = self.attention(hidden, enc_output)
        y = self.embedding(y)
        y = tf.concat([tf.expand_dims(context_vector, 1), y], axis=-1)
        output, state = self.gru(y)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.fc(output)
        return y, state, attention_weights

    def initialize_hidden_state(self):
        """
        Initializes the hidden state.

        Returns:
        - hidden_state: A tensor of zeros with shape (batch_size, dec_units).
        """
        return tf.zeros((self.batch_size, self.dec_units))
