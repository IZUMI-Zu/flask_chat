import tensorflow as tf

class Encoder(tf.keras.Model):
    """
    Encoder module for sequence-to-sequence models.

    This class implements the encoder module for sequence-to-sequence models. It takes an input sequence,
    an initial hidden state, and produces the encoder output and the final hidden state.

    Args:
    - vocab_size (int): The size of the vocabulary.
    - embedding_dim (int): The dimensionality of the embedding layer.
    - enc_units (int): The number of units in the encoder GRU layer.
    - batch_size (int): The batch size.

    Methods:
    - call(x, hidden): Performs the forward pass of the encoder.
      - x: The input sequence.
      - hidden: The initial hidden state.
      Returns the encoder output and the final hidden state.

    Note:
    - This implementation uses the GRU (Gated Recurrent Unit) as the encoder RNN.
    """
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        """
        Performs the forward pass of the encoder.

        Args:
        - x: The input sequence.
        - hidden: The initial hidden state.

        Returns:
        - output: The encoder output.
        - state: The final hidden state.

        The encoder takes an input sequence and an initial hidden state as inputs.
        It first embeds the input sequence and then passes it through the GRU layer.
        The GRU layer returns the output sequence and the final hidden state.

        Note:
        - The input sequence and the initial hidden state should have compatible shapes for matrix multiplication.
        """
        x_emb = self.embedding(x)
        output, state = self.gru(x_emb, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        """
        Initializes the hidden state.

        Returns:
        - hidden_state: A tensor of zeros with shape (batch_size, enc_units).
        """
        return tf.zeros((self.batch_size, self.enc_units))
