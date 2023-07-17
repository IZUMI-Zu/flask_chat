#!/usr/bin/env python
# coding: utf-8

# In[6]:


#切换cpu执行
import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[7]:


#导入工具库
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[8]:


#查看tf版本
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("Tensorflow version : ",tf.__version__)


# In[9]:


#数据集读取和预处理
dataset = pd.read_csv("weibo_senti_100k.csv",engine="python",header=None)
dataset.head()


# In[10]:


# 删除第0行数据
dataset= dataset.drop([0])
dataset.columns = ['label','text']
dataset.head()


# In[11]:


def read_txt(filepath):
    file = open(filepath,'r',encoding='utf-8')
    txt=[line.strip()for line in file]#移除字符串头尾指定的字符
    file.close()
#     txt=file.read()
    return txt


# In[12]:


import re
import jieba

# 匹配[^\u4e00-\u9fa5]中文字符

def select_Chinese(text):
    pattern =re.compile(r'[^\u4e00-\u9fa5]')
    chinesetext=re.sub(pattern,'',str(text)).strip()
    return chinesetext
    
# 中文分词并去除停用词

def pro_sentence(sentence):
#   中文分词
    sentence_seged=jieba.cut(sentence.strip())
#   读取停用词表
    stopwords = read_txt('hit_stopwords.txt')
    protxt=""
    for word in sentence_seged:
        if word not in stopwords:
            if word !='/t':
                protxt+=word
                protxt+=" "
    return protxt    


# In[13]:


# 对数据集中的text列中每行文本进行预处理
dataset.text = dataset.text.apply(lambda x : select_Chinese(x))
dataset.text = dataset.text.apply(lambda x : pro_sentence(x))


# In[14]:


dataset.text[2]


# In[15]:


MAX_WORDS = 100000        #最大词汇量10万
MAX_SEQ_LENGTH = 30       #最大序列长度


# In[16]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[17]:


#标签类别进行LabelEncoding,将类别编码成连续的编号
from sklearn.preprocessing import LabelEncoder                  
encoder = LabelEncoder()


# In[18]:


EMBEDDING_DIM = 300 #300维
BATCH_SIZE = 1000 #批处理大小
EPOCHS = 5 #循环次数
LR = 1e-3 #学习率
MODEL_PATH = "./training_model.hdf5" #保存模型路径


# In[19]:


#去除标点
def split_sentence(sentence):
    stop ='[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    sentence=re.sub(stop,'',sentence)
    return sentence.split()
data1 = dataset
sentences = data1.text.apply(split_sentence)
print(sentences)


# In[20]:


import gensim
from gensim.models import word2vec
#sg=1采用skip-gram算法
model = gensim.models.Word2Vec(sentences=sentences,vector_size =EMBEDDING_DIM,sg=1,min_count=5,window=5)


# In[21]:


#字向量保存
model.wv.save_word2vec_format('word_data.vector',binary=False)
#模型保存
model.save('worddd.model')


# In[22]:


#生成词表
vocab_list=list(model.wv.index_to_key)
word_index = {word:index for index,word in enumerate(vocab_list)}
def get_index(sentence):
    global word_index
    sequence = []
    for word in sentence:
        try:
            sequence.append(word_index[word])
        except KeyError:
            pass
    return sequence
X_data = list(map(get_index,sentences))


# In[23]:


vocabsize=len(word_index)
vocabsize


# In[24]:


print(word_index)


# In[25]:


#序列化+填充
X_pad = pad_sequences(X_data,maxlen=MAX_SEQ_LENGTH)
Y=encoder.fit_transform(data1.label.tolist())
X_train,X_test,Y_train,Y_test = train_test_split(X_pad,Y,test_size=0.2,random_state=42,shuffle=True)
# X_train,X_test = train_test_split(X_pad,test_size=0.2,random_state=666,shuffle=True)
print(X_pad)


# In[26]:


#获取词嵌入矩阵
embedding_matrix1 = model.wv.vectors


# In[27]:


#训练模型层
from tensorflow.keras.layers import Conv1D,Bidirectional,LSTM,Dense,Input,Dropout,Embedding
from tensorflow.keras.layers import SpatialDropout1D #丢弃整个1D的特征图而不是丢弃单个神经元，提高特征图之间的独立性
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten


mod = Sequential()
mod.add( Embedding(input_dim=embedding_matrix1.shape[0],
                     output_dim=embedding_matrix1.shape[1],
                     input_length=MAX_SEQ_LENGTH,
                     weights=[embedding_matrix1],
                     trainable=False))
mod.add(SpatialDropout1D(0.2))            
mod.add(Conv1D(64,5,activation='relu'))  #卷积层
mod.add(Bidirectional(LSTM(64,dropout=0.2,recurrent_dropout=0.2)))
mod.add(Dense(512,activation='relu'))  #全连接层
mod.add(Dropout(0.5))
mod.add(Dense(512,activation='relu'))
mod.add(Dense(1,activation='sigmoid'))


# In[28]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

#模型编译
mod.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=LR),
              metrics=['accuracy'])
ReduceLR = ReduceLROnPlateau(factor=0.1,min_lr=0.01,monitor='val_loss',verbose=0) #当某指标不再变化（下降或升高），调整学习率，这是非常实用的学习率调整策略。
#factor:学习速率被降低的因数，新的学习速率=学习率*因素
#min_lr:学习率的下界
#monitor:被监测的数据

history = mod.fit(x=X_train,
                    y=Y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_test,Y_test),
                    callbacks=[ReduceLR])
mod.save('first_model.h5')


# In[29]:


s,(one,two) = plt.subplots(2,1)
one.plot(history.history['accuracy'],c='b')
one.plot(history.history['val_accuracy'],c='r')
one.set_title('Model Accuracy')
one.set_ylabel("accuracy")
one.set_xlabel("epoch")
one.legend(['LSTM train','LSTM val'],loc='upper left')

two.plot(history.history['loss'],c='m')
two.plot(history.history['val_loss'],c='c')
two.set_title('Model Loss')
two.set_ylabel('loss')
two.set_xlabel('epoch')
two.legend(['train','val'],loc='upper left')
mod.save('first_model.h5')


# In[30]:


#模型的输出概率在0-1之间，设定一个阈值为0.5，概率超过0.5判定为正面评论，否则为负面评论
def judge(score):
    return 1 if score > 0.5 else 0


# In[31]:


#模型在test上预测
scores = mod.predict(X_test,verbose=1,batch_size=1000)
#最终的预测结果
Y_pred = [judge(score) for score in scores]
Y_test.squeeze()


# In[32]:


#模型性能评估
from sklearn.metrics import confusion_matrix
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.subplots as tls
import plotly.figure_factory as ff
def show_matrics(y_test,y_pred):
    #计算混淆矩阵
    conf_matrix = confusion_matrix(Y_test,Y_pred)
    trace1 = go.Heatmap(z=conf_matrix,x=["0(pred)","1(pred)"],
                       y=["0(True)","1(True)"],xgap = 2,ygap = 2,
                       colorscale = 'Viridis',showscale = False)
    #根据混淆矩阵，获取对应的参数值
    TP = conf_matrix[1,1]
    FN = conf_matrix[1,0]
    FP = conf_matrix[0,1]
    TN = conf_matrix[0,0]
    
    #计算accuracy,precision,recall,fl_score
    accuracy=(TP+TN)/(TP+TN+FP+FN)#准确率
    precision = TP/(TP+FP)#精确率
    recall = TP/(TP+FN)#召回率
    Fl_score = 2 * precision * recall/(precision + recall) #F1分数
    #显示以上四个指标
    show_metrics = pd.DataFrame(data=[[accuracy,precision,recall,Fl_score]])
    show_metrics = show_metrics.T
    #可视化显示
    colors = ['gold','lightgreen','lightcoral','lightskyblue']
    trace2 = go.Bar(x=show_metrics[0].values,
                   y=['Accuracy','Precision','Recall','Fl_score'],
                   text = np.round_(show_metrics[0].values,4),
                   textposition='auto',
                   orientation='h',
                   opacity=0.8,
                   marker=dict(color=colors,line=dict(color="#000000",width=1.5)))
    fig=tls.make_subplots(rows=2,cols=1,subplot_titles=('Confusion Matrix','Metrics'))
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,2,1)
    py.iplot(fig)

show_matrics(Y_test.squeeze(),Y_pred)


# In[33]:


#文本情感预测
# text_ori ="这部电影真的太难看了，两个钟头简直就是浪费时间，太让我伤心了。"
# text_ori ="w我好喜欢这部电影，剧情跌宕，角色塑造丰满，电影票买的太值了！"
# text_ori="天气真冷，心情糟糕透了，好难过！"
# text_ori="天气晴朗，万物可爱，今天真是充满活力的一天啊！"
text_pre1=select_Chinese(text_ori)
text_pre2=pro_sentence(text_pre1)
stop ='[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
text = re.sub(stop,'',text_pre2).split()
sentence1=[]
sentence1.append(text)
testdata=list(map(get_index,sentence1))
testdata_fin=pad_sequences(testdata,maxlen=MAX_SEQ_LENGTH)
print(testdata_fin)
scores = mod.predict(testdata_fin)
pred_score=judge(scores)
if(pred_score==1):
    print("正面文本，分数是： ",scores)
else:
    print("负面文本，分数是： ",scores)
    


# In[ ]:


#文本相似度匹配
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


def wordaver(model,words):
    return np.mean([model.wv.get_vector(word) for word in words],axis=0)
#测试不同阈值临界，比较不同数值的优劣
def mark_score(score):
    if score<=0.6:
        print("文本相似度低")
    elif score<=0.75:
        print("文本相似度中")
    else:
        print("文本相似度高")
    


# In[ ]:


s1="我喜欢他，看到他好激动哈哈，爱死他了！"
s2="我真的太讨厌他了，真的让人恶心。"
# s1="天气很差，外面很冷，浑身难受。"
# s2="天气真冷，心情糟糕透了，好难过！"
# s1="这部电影真好看，强烈推荐。"
# s2="看过电影了，真的很不错。"

s1_pre1=select_Chinese(s1)
s1_pre2=pro_sentence(s1_pre1)
s1_fin=split_sentence(s1_pre2)
s2_pre1=select_Chinese(s2)
s2_pre2=pro_sentence(s2_pre1)
s2_fin=split_sentence(s2_pre2)
print(s1_fin)
print(s2_fin)


# In[ ]:


s1_aver =wordaver(model,s1_fin)#s1的向量表示
s2_aver =wordaver(model,s2_fin)#s2的向量表示


# In[ ]:


#比较两段文本余弦相似度
print(cosine_similarity(s1_aver.reshape(1,-1),s2_aver.reshape(1,-1)))
mark_score(cosine_similarity(s1_aver.reshape(1,-1),s2_aver.reshape(1,-1)))


# In[ ]:


from builtins import bytes, range
 
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.manifold import TSNE
import gensim
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# font = FontProperties(fname="himalaya.ttf",size=20)
from random import sample

plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['font.size’] = 10
plt.rcParams['axes.unicode_minus']=False 


# In[ ]:


def tsne_plot(model, words_num):
 
    labels = []
    tokens = []
    for word in model.wv.index_to_key:
        tokens.append(model.wv[word])
        labels.append(word)
    tsne_model = TSNE(perplexity=30, n_components=3, init='pca', n_iter=1000, random_state=23)
    new_values = tsne_model.fit_transform(np.array(tokens))
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(10, 10))
    for i in range(words_num):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# In[ ]:


model_pic=gensim.models.Word2Vec.load('worddd.model')
tsne_plot(model_pic, 200)


# In[34]:


#chatbot功能实现
#文本预处理
from zhon.hanzi import punctuation

convs = open('protext.txt','w',encoding='utf8')  # conversation set
with open('xiaohuangji50w_nofenci.conv', encoding="utf8") as f:
    one_conv = ""  # a complete conversation
    for line in f:
        line = line.strip('\n')
        line = re.sub(r"[%s]+" % punctuation,"",line)#去除标点
        line = re.sub('[\d]','',line) # 去除数字[0-9]
        if line == '':
            continue
        if line[0] == 'E':
            if one_conv:
                convs.write(one_conv[:-1]+'\n')
            one_conv = ""
        elif line[0] == 'M':
            one_conv=one_conv+str(" ".join(jieba.cut(line.split(' ')[1])))+ '\t'


# In[ ]:


print(one_conv)


# In[ ]:


def create_vocab(lang, vocab_path, vocab_size):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
#     tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(lang)
    vocab = json.loads(tokenizer.to_json(ensure_ascii=False))
    vocab['index_word'] = tokenizer.index_word
    vocab['word_index'] = tokenizer.word_index
    vocab['document_count']=tokenizer.document_count
    vocab = json.dumps(vocab, ensure_ascii=False)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write(vocab)
    f.close()
    print("字典保存在:{}".format(vocab_path))
    
    
def preprocess_sentence(w):
    w = 'start ' + w + ' end'
    return w


# In[ ]:


import io
import json
import time
import sys


# In[ ]:


lines = open('protext.txt', encoding='utf8').readlines()
word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines]
input_lang, target_lang = zip(*word_pairs)

create_vocab(input_lang,"inp.vocab",20000)
create_vocab(target_lang,"tar.vocab",20000)


# In[ ]:


vocab_inp_size = 20000
vocab_tar_size = 20000
embedding_dim = 128
units = 256
batch_size = 128


# In[ ]:


#模型构建

# 定义Encoder类
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


# In[ ]:


# 定义bahdanauAttention是常用的attention实现方法之一
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


# In[ ]:


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


# In[ ]:


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # mask掉start,去除start对于loss的干扰
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)  # 将bool型转换成数值
    loss_ *= mask
    return tf.reduce_mean(loss_)


# In[ ]:


# 实例化encoder、decoder、optimizer、checkpoint等
encoder1 = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
decoder1 = Decoder(vocab_tar_size, embedding_dim, units, batch_size)
optimizer1 = tf.keras.optimizers.Adam()
checkpoint1 = tf.train.Checkpoint(optimizer=optimizer1, encoder=encoder1, decoder=decoder1)


# In[ ]:


def training_step(inp, targ, targ_lang, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder1(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['start']] * BATCH_SIZE, 1)
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder1(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)
    step_loss = (loss / int(targ.shape[1]))
    variables = encoder1.trainable_variables + decoder1.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer1.apply_gradients(zip(gradients, variables))
    return step_loss


# In[ ]:


def tokenize(vocab_file):
    #从词典中读取预先生成tokenizer的config，构建词典矩阵
    with open(vocab_file,'r',encoding='utf-8') as f:
        tokenize_config=json.dumps(json.load(f),ensure_ascii=False)
        lang_tokenizer=tf.keras.preprocessing.text.tokenizer_from_json(tokenize_config)
    #利用词典进行word2number的转换以及padding处理
    return lang_tokenizer


# In[ ]:


#序列化
input_tokenizer=tokenize("inp.vocab")
target_tokenizer=tokenize("tar.vocab")

input_tensor=input_tokenizer.texts_to_sequences(input_lang)
target_tensor=target_tokenizer.texts_to_sequences(target_lang)

input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=20,
                                                           padding='post')
target_tensor= tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=20,
                                                           padding='post')


# In[ ]:


steps_per_epoch = len(input_tensor) // 128
BUFFER_SIZE = len(input_tensor)
dataset = tf.data.Dataset.from_tensor_slices((input_tensor,target_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
enc_hidden = encoder1.initialize_hidden_state()
writer = tf.summary.create_file_writer("log_dir")


# In[ ]:


# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()     #迁移1.x ->2.x
# sess = tf.Session()


# In[ ]:


#定义训练函数
def train():

    # 从训练语料中读取数据并使用预生成词典word2number的转换
    print("Preparing data in %s" % "train_data")
    print('每个epoch的训练步数: {}'.format(steps_per_epoch))
    #如有已经有预训练的模型则加载预训练模型继续训练
    checkpoint_dir = 'model_data'
    ckpt=tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        print("reload pretrained model")
        checkpoint1.restore(tf.train.latest_checkpoint(checkpoint_dir))

    #使用Dataset加载训练数据，Dataset可以加速数据的并发读取并进行训练效率的优化
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    start_time = time.time()
    #current_loss=2
    #min_loss=gConfig['min_loss']
    epoch = 0
    train_epoch = 5
    #开始进行循环训练，这里设置了一个结束循环的条件就是当loss小于设置的min_loss超参时终止训练
    while epoch<train_epoch:
        start_time_epoch = time.time()
        total_loss = 0
        #进行一个epoch的训练，训练的步数为steps_per_epoch
        for batch,(inp, targ) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = training_step(inp, targ,target_tokenizer, enc_hidden)
            total_loss += batch_loss
            print('epoch:{}batch:{} batch_loss: {}'.format(epoch,batch,batch_loss))
        #结束一个epoch的训练后，更新current_loss，计算在本epoch中每步训练平均耗时、loss值
        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        current_steps = +steps_per_epoch
        epoch_time_total = (time.time() - start_time)
        print('训练总步数: {} 总耗时: {}  epoch平均每步耗时: {} 平均每步loss {:.4f}'
              .format(current_steps, epoch_time_total, step_time_epoch, step_loss))
        #将本epoch训练的模型进行保存，更新模型文件
        checkpoint1.save(file_prefix=checkpoint_prefix)
        sys.stdout.flush()
        epoch = epoch + 1
        with writer.as_default():
            tf.summary.scalar('loss', step_loss, step=epoch)


# In[ ]:


def predict(sentence):
    # 从词典中读取预先生成tokenizer的config，构建词典矩阵
    input_tokenizer = tokenize("inp.vocab")
    target_tokenizer = tokenize("tar.vocab")
    #加载预训练的模型
    checkpoint_dir = 'model_data'
    checkpoint1.restore(tf.train.latest_checkpoint(checkpoint_dir))
    #对输入的语句进行处理，加上start end标示
    sentence = preprocess_sentence(sentence)
#     print(sentence)
    inputs = input_tokenizer.texts_to_sequences([sentence])[0]
    #进行padding的补全
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],maxlen=20,padding='post')
    inputs = tf.convert_to_tensor(inputs)
#     print(inputs)
    result = ''
    #初始化一个中间状态
    hidden = [tf.zeros((1, 256))]
    #对输入上文进行encoder编码，提取特征
    enc_out, enc_hidden = encoder1(inputs, hidden)
    dec_hidden = enc_hidden
    
    #decoder的输入从start的对应Id开始正向输入
    dec_input = tf.expand_dims([target_tokenizer.word_index['start']], 0)
#     print(dec_input)

    #在最大的语句容长度范围内，使用模型中的decoder进行循环解码
    for t in range(4):
        #获得解码结果，并使用argmax确定概率最大的id
        predictions, dec_hidden, attention_weights =decoder1(dec_input, dec_hidden, enc_out)
#         print(predictions)
        predicted_id = tf.argmax(predictions[0]).numpy()+1
#         print(predicted_id)
#         print(target_tokenizer.index_word[predicted_id])
        #判断当前Id是否为语句结束表示，如果是则停止循环解码，否则进行number2word的转换，并进行语句拼接
        if target_tokenizer.index_word[predicted_id] == 'end':
            break
        result += str(target_tokenizer.index_word[predicted_id]) + ' '
        #将预测得到的id作为下一个时刻的decoder的输入
        dec_input = tf.expand_dims([predicted_id], 0)
    return result


# In[ ]:


# train()


# In[ ]:


predict('你 主人 是 谁 呀')


# In[ ]:


predict('晚安')


# In[ ]:


predict('我 爱 你')


# In[ ]:


predict('滚')


# In[ ]:


predict('你 好')


# In[ ]:


predict('离 我 远 点')


# In[ ]:


predict('你 是 好人 吗')


# In[ ]:


predict('你 是 男生 吗')


# In[ ]:


# predict('你 好')


# In[ ]:




