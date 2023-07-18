import tensorflow as tf

from chat_box.bahdanau_attention import BahdanauAttention

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        # 初始化batch_sz、dec_units、embedding 、gru 、fc、attention
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
        # 首先对enc_output、以及decoder的hidden计算attention，输出上下文语境向量
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # 对decoder的输入进行embedding
        y = self.embedding(y)
        # 拼接上下文语境与decoder的输入embedding，并送入gru中
        y = tf.concat([tf.expand_dims(context_vector, 1), y], axis=-1)
        output, state = self.gru(y)
        # 将gru的输出进行维度转换，送入全连接神经网络 得到最后的结果
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.fc(output)
        return y, state, attention_weights
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.dec_units))