# 先不考虑文本过长的问题
# ! -*- coding: utf-8 -*-

import codecs
import csv
import gc
import os
import pickle
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import *
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_bert.backend import keras
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from tqdm import tqdm
import os, re, csv
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
import gc
from random import choice

import tensorflow as tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

maxlen = 256  # 140
learning_rate = 5e-5  # 5e-5
min_learning_rate = 1e-5  # 1e-5
bsize = 16
bert_kind = ['chinese_L-12_H-768_A-12', 'tf-bert_wwm_ext'][0]
config_path = os.path.join('../bert', bert_kind, 'bert_config.json')
checkpoint_path = os.path.join('../bert', bert_kind, 'bert_model.ckpt')
dict_path = os.path.join('../bert', bert_kind, 'vocab.txt')

model_save_path = "./data/ccks2020_ckt_256/"
train_data_path = '../model1/data/event_entity_train_data_label.csv'
test_data_path = '../model1/data/event_entity_dev_data.csv'
sep = '\t'

if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            elif c == 'S':
                R.append('[unused2]')
            elif c == 'T':
                R.append('[unused3]')
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


def delete_tag(s):
    s = re.sub('\{IMG:.?.?.?\}', '', s)  # 图片
    s = re.sub(re.compile(r'[a-zA-Z]+://[^\s]+'), '', s)  # 网址
    s = re.sub(re.compile('<.*?>'), '', s)  # 网页标签
    s = re.sub(re.compile('&[a-zA-Z]+;?'), '', s)  # 网页标签
    s = re.sub(re.compile('[a-zA-Z0-9]*[./]+[a-zA-Z0-9./]+[a-zA-Z0-9./]*'), '', s)
    s = re.sub("\?{2,}", "", s)
    # s = re.sub("（", ",", s)
    # s = re.sub("）", ",", s)
    s = re.sub(" \(", "（", s)
    s = re.sub("\) ", "）", s)
    s = re.sub("\u3000", "", s)
    # s = re.sub(" ", "", s)
    r4 = re.compile('\d{4}[-/年](\d{2}([-/月]\d{2}[日]{0,1}){0,1}){0,1}')  # 日期
    s = re.sub(r4, "▲", s)
    return s


# 读取训练集
data = pd.read_csv(train_data_path, encoding='utf-8', sep=sep,
                   names=['id', 'text', 'Q', 'A'], quoting=csv.QUOTE_NONE)

# data["text"] = data["text"].map(lambda x: x[1:-1])
# data["Q"] = data["Q"].map(lambda x: x[1:-1])
# data["A"] = data["A"].map(lambda x: x[1:-1])

data['text'] = [delete_tag(s) for s in data.text]
# NaN替换成'NaN'
data.fillna('NaN', inplace=True)
data = data[data.A != 'NaN']

classes = set(data["Q"].unique())

entity_train = list(set(data['A'].values.tolist()))

# ClearData
data.drop("id", axis=1, inplace=True)  # drop id
data.drop_duplicates(['text', 'Q', 'A'], keep='first', inplace=True)  # drop duplicates
data.drop("Q", axis=1, inplace=True)  # drop Q

data["A"] = data["A"].map(lambda x: str(x).replace('NaN', ''))

data["e"] = data.apply(lambda row: 1 if row['A'] in row['text'] else 0, axis=1)

data = data[data["e"] == 1]
data = data.groupby(['text'], sort=False)['A'].apply(lambda x: ';'.join(x)).reset_index()

train_data = []
for t, n in zip(data["text"], data["A"]):
    train_data.append((t, n))
print('最终训练集大小:%d' % len(train_data))
print('-' * 30)

D = pd.read_csv(test_data_path, header=None, sep=sep,
                names=["id", "text", "event"], quoting=csv.QUOTE_NONE)
D['text'] = [delete_tag(s) for s in D.text]
D.fillna('NaN', inplace=True)
# D['event'] = D['event'].map(lambda x: "公司股市异常" if x == "股市异常" else x)
D['text'] = D['text'].map(
    lambda x: x.replace("\x07", "").replace("\x05", "").replace("\x08", "").replace("\x06", "").replace("\x04", ""))

test_data = []
for id, t in zip(D["id"], D["text"]):
    test_data.append((id, t))
print('最终测试集大小:%d' % len(test_data))
print('-' * 30)

BIOtag = ['O', 'B', 'I']
tag2id = {v: i for i, v in enumerate(BIOtag)}


def getBIO(text, e):
    text = text[:maxlen]
    x1 = tokenizer.tokenize(text)
    p1 = [0] * len(x1)
    # print(text,e)
    for ei in e.split(';'):
        if ei == '':
            continue
        x2 = tokenizer.tokenize(ei)[1:-1]
        # print(x2)
        for i in range(len(x1) - len(x2)):
            if x2 == x1[i:i + len(x2)] and sum(p1[i:i + len(x2)]) == 0:
                pei = [tag2id['I']] * len(x2)
                pei[0] = tag2id['B']
                p1[i:i + len(x2)] = pei

    maxN = len(BIOtag)
    id2matrix = lambda i: [1 if x == i else 0 for x in range(maxN)]
    p1 = [id2matrix(i) for i in p1]

    return p1


def seq_padding(X, padding=0, wd=1):
    L = [len(x) for x in X]
    ML = max(L)  # maxlen
    if wd == 1:
        return np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])
    else:
        return np.array([
            np.concatenate([x, [[padding] * len(x[0])] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])


class data_generator:
    def __init__(self, data, batch_size=bsize):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, P = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                e = d[1]
                # todo 构造标签
                p = getBIO(text, e)
                x1, x2 = tokenizer.encode(text)
                X1.append(x1)
                X2.append(x2)
                P.append(p)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    P = seq_padding(P, wd=2)
                    # print(X1.shape)
                    # print(P.shape)
                    yield [X1, X2], P
                    X1, X2, P = [], [], []


# 定义模型

from keras import backend as K


def myloss(y_true, y_pred):
    return K.mean(K.sum(K.categorical_crossentropy(y_true, y_pred, axis=-1, from_logits=False), axis=-1))


def modify_bert_model_3():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    bert_out = bert_model([x1, x2])  # [batch,maxL,768]
    # todo [batch,maxL,768] -》[batch,maxL,3]
    xlen = 1
    a = Dense(units=xlen, use_bias=False, activation='tanh')(bert_out)  # [batch,maxL,1]
    b = Dense(units=xlen, use_bias=False, activation='tanh')(bert_out)
    c = Dense(units=xlen, use_bias=False, activation='tanh')(bert_out)
    outputs = Lambda(lambda x: K.concatenate(x, axis=-1))([a, b, c])  # [batch,maxL,3]
    outputs = Softmax()(outputs)
    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_model')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    return model


def decode(text_in, p_in):
    '''解码函数'''
    p = np.argmax(p_in, axis=-1)
    # _tokens = tokenizer.tokenize(text_in)
    _tokens = ' %s ' % text_in
    ei = ''
    r = []
    for i, v in enumerate(p):
        if ei == '':
            if v == tag2id['B']:
                ei += _tokens[i]
        else:
            if v == tag2id['B']:
                r.append(ei)
                ei = _tokens[i]
            elif v == tag2id['I']:
                ei += _tokens[i]
            elif v == tag2id['O']:
                r.append(ei)
                ei = ''
    r = [i for i in r if len(i) > 1]
    r = set(r)
    return ';'.join(r)


def extract_entity(text_in):
    """解码函数，应自行添加更多规则，保证解码出来的是一个公司名
    """
    text_in = text_in[:maxlen]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    _p = model.predict([_x1, _x2])[0]
    a = decode(text_in, _p)
    return a


def myF1_P_R(y_true, y_pre):
    a = set(y_true.split(';'))
    b = set(y_pre.split(';'))
    TP = len(a & b)
    FN = len(a - b)
    FP = len(b - a)
    P = TP / (TP + FP) if TP + FP != 0 else 0
    R = TP / (TP + FN) if TP + FN != 0 else 0
    F1 = 2 * P * R / (P + R) if P + R != 0 else 0

    return F1, P, R


def evaluate(dev_data):
    A = 1e-10
    F = 1e-10
    for d in tqdm(iter(dev_data)):
        text = d[0][:maxlen]
        e = d[1]
        y = extract_entity(d[0])
        Y = e
        if type(e) != str:
            Y = decode(text, e)
        f, p, r = myF1_P_R(Y, y)
        A += p
        F += f
    return A / len(dev_data), F / len(dev_data)


class Evaluate(Callback):
    def __init__(self, dev_data, model_path):
        self.ACC = []
        self.best = 0.
        self.passed = 0
        self.dev_data = dev_data
        self.model_path = model_path

    # 调整学习率？todo
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(self.dev_data)[0]
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            print("save best model weights ...")
            model.save_weights(self.model_path)
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))


def test(test_data, result_path):
    F = open(result_path, 'w', encoding='utf-8')
    for d in tqdm(iter(test_data)):
        s = u'%s\t%s\t%s\n' % (d[0], d[1], extract_entity(d[1]))
        F.write(s)
    F.close()


# Model
flodnums = 10

# 拆分验证集
cv_path = os.path.join(model_save_path, 'cv.pkl')
if not os.path.exists(cv_path):
    kf = KFold(n_splits=flodnums, shuffle=True, random_state=520).split(train_data)
    # save
    save_kf = []
    for i, (train_fold, test_fold) in enumerate(kf):
        save_kf.append((train_fold, test_fold))
    f = open(cv_path, 'wb')
    pickle.dump(save_kf, f, 4)
    f.close()
    kf = save_kf
else:
    f = open(cv_path, 'rb')
    kf = pickle.load(f)
    f.close()

score = []

for i, (train_fold, test_fold) in enumerate(kf):
    print("kFlod ", i, "/", flodnums)
    train_ = [train_data[i] for i in train_fold]
    dev_ = [train_data[i] for i in test_fold]

    model = modify_bert_model_3()

    train_D = data_generator(train_)
    dev_D = data_generator(dev_)

    model_path = os.path.join(model_save_path, "modify_bert_model" + str(i) + ".weights")
    if not os.path.exists(model_path):
        evaluator = Evaluate(dev_, model_path)
        model.fit_generator(train_D.__iter__(),
                            steps_per_epoch=len(train_D),
                            epochs=10,
                            callbacks=[evaluator],
                            validation_data=dev_D.__iter__(),
                            validation_steps=len(dev_D)
                            )

    print("load best model weights ...")
    model.load_weights(model_path)

    print('val')
    score.append(evaluate(dev_))
    print("valid evluation:", score[-1])
    print("valid score:", score)
    print("valid mean score:", np.mean(score))
    print('test')
    result_path = os.path.join(model_save_path, "result_k" + str(i) + ".txt")
    test(test_data, result_path)

    gc.collect()
    del model
    gc.collect()
    K.clear_session()
    a = 0 / 0
