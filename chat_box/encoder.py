import tensorflow as tf

class Encoder(tf.keras.Model):
    # 初始化函数，对默认参数进行初始化
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')

    # 定义调用函数，实现逻辑计算
    def call(self, x, hidden):
        x_emb = self.embedding(x)
        output, state = self.gru(x_emb, initial_state=hidden)
        return output, state
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))