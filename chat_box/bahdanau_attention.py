import tensorflow as tf

class BahdanauAttention(tf.keras.Model):
    """
    Bahdanau Attention mechanism for sequence-to-sequence models.

    This class implements the Bahdanau attention mechanism, which is used to compute attention weights
    and create a context vector for sequence-to-sequence models. It takes a query vector and a set of
    values (encoder outputs) as inputs and produces attention weights and a context vector.

    Args:
    - units (int): The number of units in the attention layer.

    Methods:
    - call(query, values): Performs the forward pass of the attention mechanism.
      - query: The query vector.
      - values: The values (encoder outputs).
      Returns the context vector and attention weights.

    Note:
    - This implementation uses the additive (Bahdanau) attention mechanism.
    """
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        """
        Performs the forward pass of the attention mechanism.

        Args:
        - query: The query vector.
        - values: The values (encoder outputs).

        Returns:
        - context_vector: The context vector.
        - attention_weights: The attention weights.

        The attention mechanism computes attention weights by applying a series of transformations
        to the query and values. It then computes a context vector by multiplying the attention weights
        with the values and taking their sum.

        The attention weights are computed using the following steps:
        1. Expand the dimensions of the query vector to match the shape of the values.
        2. Apply linear transformations to the values and the expanded query vector.
        3. Apply the tanh activation function to the sum of the linear transformations.
        4. Apply another linear transformation to obtain a one-dimensional score for each value.
        5. Apply the softmax function to the scores to obtain attention weights.

        The context vector is computed by element-wise multiplication of the attention weights and the values,
        followed by summing along the time axis.

        Note:
        - The query and values should have compatible shapes for matrix multiplication.
        """
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
