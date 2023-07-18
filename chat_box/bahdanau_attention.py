import tensorflow as tf

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # 注意力网络的初始化
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # 将query增加一个维度，以便可以与values进行线性相加
        hidden_with_time_axis = tf.expand_dims(query, 1)
        # 将quales与hidden_with_time_axis进行线性相加后，使用tanh进行非线性变换，最后输出一维的score
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        # 使用softmax将score进行概率化转换，转为为概率空间
        attention_weights = tf.nn.softmax(score, axis=1)
        # 将权重与values（encoder_out)进行相乘，得到context_vector
        context_vector = attention_weights * values
        # 将乘积后的context_vector按行相加，进行压缩得到最终的context_vector
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights