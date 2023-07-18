import json
import tensorflow as tf

from chat_box.bahdanau_attention import BahdanauAttention
from chat_box.decoder import Decoder
from chat_box.encoder import Encoder

class ModelPredictor:
    def __init__(self):
        pass

    def tokenize(self, vocab):
        with open(vocab,'r',encoding='utf-8') as f:
            tokenize_config = json.dumps(json.load(f),ensure_ascii=False)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenize_config)
        # 利用词典进行word2number的转换以及padding处理
        return tokenizer

    def preprocess_sentence(self, sentence):
        sentence = 'start ' + sentence + ' end'
        return sentence

    def predict(self, sentence):

        vocab_inp_size = 20000
        vocab_tar_size = 20000
        embedding_dim = 128
        units = 256
        batch_size = 1000

        encoder1 = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
        decoder1 = Decoder(vocab_tar_size, embedding_dim, units, batch_size)
        optimizer1 = tf.keras.optimizers.Adam()
        checkpoint1 = tf.train.Checkpoint(optimizer=optimizer1, encoder=encoder1, decoder=decoder1)

        # 从词典中读取预先生成tokenizer的config，构建词典矩阵
        input_tokenizer = self.tokenize("model/inp.vocab")
        target_tokenizer = self.tokenize("model/tar.vocab")
        #加载预训练的模型
        checkpoint_dir = 'model/model_data'
        checkpoint1.restore(tf.train.latest_checkpoint(checkpoint_dir))
        #对输入的语句进行处理，加上start end标示
        sentence = self.preprocess_sentence(sentence)
        
        inputs = input_tokenizer.texts_to_sequences([sentence])[0]
        #进行padding的补全
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],maxlen=20,padding='post')
        inputs = tf.convert_to_tensor(inputs)
        result = ''
        #初始化一个中间状态
        hidden = [tf.zeros((1, 256))]
        #对输入上文进行encoder编码，提取特征
        enc_out, enc_hidden = encoder1(inputs, hidden)
        dec_hidden = enc_hidden
    
        #decoder的输入从start的对应Id开始正向输入
        dec_input = tf.expand_dims([target_tokenizer.word_index['start']], 0)
        #在最大的语句容长度范围内，使用模型中的decoder进行循环解码
        for t in range(4):
            #获得解码结果，并使用argmax确定概率最大的id
            predictions, dec_hidden, attention_weights =decoder1(dec_input, dec_hidden, enc_out)
            predicted_id = tf.argmax(predictions[0]).numpy()+1
            #判断当前Id是否为语句结束表示，如果是则停止循环解码，否则进行number2word的转换，并进行语句拼接
            if target_tokenizer.index_word[predicted_id] == 'end':
                break
            result += str(target_tokenizer.index_word[predicted_id]) + ' '
            #将预测得到的id作为下一个时刻的decoder的输入
            dec_input = tf.expand_dims([predicted_id], 0)
        return result
