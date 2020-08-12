# ! -*- coding: utf-8 -*-

from keras import backend as K
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
from keras_contrib.layers import CRF
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from tqdm import tqdm
import os
import re
import csv
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
import gc
from random import choice
# 引入Tensorboard
from keras.callbacks import TensorBoard
import tensorflow as tf
from model2.GRU_model import MyGRU

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
flodnums = 5
tagkind = ['BIO', 'BIOES'][1]

maxlen = 256  # 140
learning_rate = 5e-5  # 5e-5
min_learning_rate = 1e-5  # 1e-5
bsize = 16
bert_kind = ['chinese_L-12_H-768_A-12', 'tf-bert_wwm_ext'][1]
config_path = os.path.join('../bert', bert_kind, 'bert_config.json')
checkpoint_path = os.path.join('../bert', bert_kind, 'bert_model.ckpt')
dict_path = os.path.join('../bert', bert_kind, 'vocab.txt')

model_cv_path = "./data/ccks2020_ckt_256_bilstm_crf_BIOES"
model_save_path = "./data/ccks2020_ckt_256_biMyGRU_crf_BIOES_300/"
train_data_path = './data/event_entity_train_data_label.csv'
test_data_path = './data/event_entity_dev_data.csv'
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
    # s = re.sub(re.compile('&[a-zA-Z]+;?'), '', s)  # 网页标签
    # s = re.sub(re.compile(
    #     '[a-zA-Z0-9]*[./]+[a-zA-Z0-9./]+[a-zA-Z0-9./]*'), '', s)
    s = re.sub("\?{2,}", "", s)
    # s = re.sub("（", ",", s)
    # s = re.sub("）", ",", s)
    s = re.sub(" \(", "（", s)
    s = re.sub("\) ", "）", s)
    s = re.sub("\u3000", "", s)
    # s = re.sub(" ", "", s)
    r4 = re.compile('\d{4}[-/年](\d{2}([-/月]\d{2}[日]{0,1}){0,1}){0,1}')  # 日期
    s = re.sub(r4, "▲", s)
    s = s.replace("\x07", "").replace("\x05", "").replace(
        "\x08", "").replace("\x06", "").replace("\x04", "")
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
dataNaN = data[data.A == 'NaN']
classes = set(data["Q"].unique())

entity_train = list(set(data['A'].values.tolist()))

# ClearData
data.drop("id", axis=1, inplace=True)  # drop id
data.drop_duplicates(['text', 'Q', 'A'], keep='first',
                     inplace=True)  # drop duplicates
data.drop("Q", axis=1, inplace=True)  # drop Q

data["A"] = data["A"].map(lambda x: str(x).replace('NaN', ''))

data["e"] = data.apply(lambda row: 1 if row['A'] in row['text'] else 0, axis=1)

data = data[data["e"] == 1]
data = data.groupby(['text'], sort=False)['A'].apply(
    lambda x: ';'.join(x)).reset_index()

train_data = []
for t, n in zip(data["text"], data["A"]):
    train_data.append((t, n))
print('最终训练集大小:%d' % len(train_data))
print('-' * 30)

D = pd.read_csv(train_data_path, header=None, sep=sep,
                names=['id', 'text', 'Q', 'A'], quoting=csv.QUOTE_NONE)
D['text'] = [delete_tag(s) for s in D.text]
D.fillna('NaN', inplace=True)
# D['event'] = D['event'].map(lambda x: "公司股市异常" if x == "股市异常" else x)

test_data = []
for id, t in zip(D["id"], D["text"]):
    test_data.append((id, t))
print('最终测试集大小:%d' % len(test_data))
print('-' * 30)
if tagkind == 'BIO':
    BIOtag = ['O', 'B', 'I']
elif tagkind == 'BIOES':
    BIOtag = ['O', 'B', 'I', 'E', 'S']
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
                if tagkind == 'BIO':
                    pei = [tag2id['I']] * len(x2)
                    pei[0] = tag2id['B']
                elif tagkind == 'BIOES':
                    pei = [tag2id['I']] * len(x2)
                    if len(x2) == 1:
                        pei[0] = tag2id['S']
                    else:
                        pei[0] = tag2id['B']
                        pei[-1] = tag2id['E']
                p1[i:i + len(x2)] = pei

    maxN = len(BIOtag)

    def id2matrix(i):
        return [1 if x == i else 0 for x in range(maxN)]

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
        padding_wd = [padding] * len(X[0][0])
        padding_wd[tag2id['O']] = 1
        return np.array([
            np.concatenate([x, [padding_wd] * (ML - len(x))]) if len(x) < ML else x for x in X
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


def myloss(y_true, y_pred):
    return K.mean(K.sum(K.categorical_crossentropy(y_true, y_pred, axis=-1, from_logits=False), axis=-1))


def modify_bert_model_3():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    bert_out = bert_model([x1, x2])  # [batch,maxL,768]
    # todo [batch,maxL,768] -》[batch,maxL,3]
    xlen = 1
    a = Dense(units=xlen, use_bias=False, activation='tanh')(
        bert_out)  # [batch,maxL,1]
    b = Dense(units=xlen, use_bias=False, activation='tanh')(bert_out)
    c = Dense(units=xlen, use_bias=False, activation='tanh')(bert_out)
    outputs = Lambda(lambda x: K.concatenate(
        x, axis=-1))([a, b, c])  # [batch,maxL,3]
    outputs = Softmax()(outputs)
    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_model')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    return model


def modify_bert_model_3_masking():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        # output_layer_num=4
    )

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]
    # Masking
    xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
    outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
    outputs = Masking(mask_value=0)(outputs)  # error ?
    outputs = Dense(units=len(BIOtag), use_bias=False,
                    activation='Softmax')(outputs)

    model = keras.models.Model(
        [x1_in, x2_in], outputs, name='basic_masking_model')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    return model


def modify_bert_model_bilstm_crf():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        output_layer_num=4
    )

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]
    # Masking
    # xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
    # outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
    # outputs = Masking(mask_value=0)(outputs)
    outputs = Bidirectional(LSTM(units=300, return_sequences=True))(outputs)
    outputs = Dropout(0.2)(outputs)
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model(
        [x1_in, x2_in], outputs, name='basic_bilstm_crf_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    return model


def modify_bert_model_biMyGRU_crf():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        output_layer_num=4
    )

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]

    outputs = Bidirectional(MyGRU(units=300, return_sequences=True,
                                  reset_after=True, name='MyGRU', tcell_num=3))(outputs)
    outputs = Dropout(0.2)(outputs)
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model(
        [x1_in, x2_in], outputs, name='basic_biMyGRU_crf_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    return model


def modify_bert_model_biGRU_crf():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        output_layer_num=4
    )

    for l in bert_model.layers:
        l.trainable = False

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]

    outputs = Bidirectional(
        GRU(units=300, return_sequences=True, reset_after=True))(outputs)
    outputs = Dropout(0.2)(outputs)
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model(
        [x1_in, x2_in], outputs, name='basic_bilstm_crf_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    return model


def modify_bert_model_3_crf():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]
    #  [batch,maxL,768] -》[batch,maxL,len(BIOtag)]

    # # Masking
    # xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
    # outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
    # outputs = Masking(mask_value=0)(outputs)

    # outputs = Dense(units=len(BIOtag), use_bias=False, activation='tanh')(outputs)  # [batch,maxL,3]
    # outputs = Lambda(lambda x: x)(outputs)
    # outputs = Softmax()(outputs)

    # crf
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_crf_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    return model


def myf1(y_true, y_pre):
    # todo
    pass


def modify_bert_model_bilstm_crf_f1():
    bert_model = load_trained_model_from_checkpoint(
        config_path, checkpoint_path,
        output_layer_num=4
    )

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入

    x1, x2 = x1_in, x2_in

    outputs = bert_model([x1, x2])  # [batch,maxL,768]
    # Masking
    # xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
    # outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
    # outputs = Masking(mask_value=0)(outputs)
    outputs = Bidirectional(LSTM(units=300, return_sequences=True))(outputs)
    outputs = Dropout(0.2)(outputs)
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model(
        [x1_in, x2_in], outputs, name='basic_bilstm_crf_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function,
                  metrics=[crf.accuracy, 'myf1'])
    model.summary()

    return model


def decode(text_in, p_in):
    '''解码函数'''
    p = np.argmax(p_in, axis=-1)
    # _tokens = tokenizer.tokenize(text_in)
    _tokens = ' %s ' % text_in
    ei = ''
    r = []
    if tagkind == 'BIO':
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
    elif tagkind == 'BIOES':
        for i, v in enumerate(p):
            if ei == '':
                if v == tag2id['B']:
                    ei = _tokens[i]
                elif v == tag2id['S']:
                    r.append(_tokens[i])
            else:
                if v == tag2id['B']:
                    ei = _tokens[i]
                elif v == tag2id['I']:
                    ei += _tokens[i]
                elif v == tag2id['E']:
                    ei += _tokens[i]
                    r.append(ei)
                    ei = ''
                elif v == tag2id['O']:
                    # r.append(ei)
                    ei = ''
                elif v == tag2id['S']:
                    r.append(_tokens[i])
                    ei = ''

    r = [i for i in r if len(i) > 1]
    r = list(set(r))
    r.sort()
    return ';'.join(r)


def extract_entity(text_in, batch=None):
    """解码函数，应自行添加更多规则，保证解码出来的是一个公司名
    """
    if batch == None:
        text_in = text_in[:maxlen]
        _tokens = tokenizer.tokenize(text_in)
        _x1, _x2 = tokenizer.encode(text_in)
        _x1, _x2 = np.array([_x1]), np.array([_x2])
        _p = model.predict([_x1, _x2])[0]
        a = decode(text_in, _p)
        return a
    else:
        text_in = [i[:maxlen] for i in text_in]
        ml = max([len(i) for i in text_in])
        x1x2 = [tokenizer.encode(i, max_len=ml) for i in text_in]
        _x1 = np.array([i[0] for i in x1x2])
        _x2 = np.array([i[1] for i in x1x2])
        _p = model.predict([_x1, _x2])
        a = []
        for i in range(len(text_in)):
            a.append(decode(text_in[i], _p[i]))
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


def evaluate(dev_data, batch=1):
    A = 1e-10
    F = 1e-10
    for idx in tqdm(range(0, len(dev_data), batch)):
        d = [i[0][:maxlen] for i in dev_data[idx:idx + batch]]
        # 真实的公司名称，形如： 公司A;公司B;公司C
        Y = [i[1] for i in dev_data[idx:idx + batch]]
        y = extract_entity(d, batch)  # 预测的公司名称，形如： 公司A;公司B

        for j in range(len(d)):
            if type(Y[j]) != str:
                Y[j] = decode(d[j], Y[j])
            f, p, r = myF1_P_R(Y[j], y[j])  # 求指标
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
            lr = (2 - (self.passed + 1.) /
                  self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(self.dev_data, bsize * 2)[0]
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            print("save best model weights ...")
            model.save_weights(self.model_path)
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))


def test(test_data, result_path, batch=1):
    F = open(result_path, 'w', encoding='utf-8')
    for idx in tqdm(range(0, len(test_data), batch)):
        d0 = [i[0] for i in test_data[idx:idx + batch]]
        d1 = [i[1] for i in test_data[idx:idx + batch]]
        y = extract_entity(d1, batch)
        for i in range(len(d0)):
            s = u'%s,%s\n' % (d0[i], y[i])
            F.write(s)
        F.flush()
    F.close()


def test_cv(test_data, batch=1):
    '''预测'''
    r = []
    for idx in tqdm(range(0, len(test_data), batch)):
        d0 = [i[0] for i in test_data[idx:idx + batch]]
        d1 = [i[1][:maxlen] for i in test_data[idx:idx + batch]]
        ml = max([len(i) for i in d1])
        x1x2 = [tokenizer.encode(i, max_len=ml) for i in d1]
        _x1 = np.array([i[0] for i in x1x2])
        _x2 = np.array([i[1] for i in x1x2])
        _p = model.predict([_x1, _x2])
        r += _p
    return r


def test_cv_decode(test_data, result, result_path):
    '''获得cv的结果'''
    result_avg = np.mean(result, axis=0)
    F = open(result_path, 'w', encoding='utf-8')
    for idx, d in enumerate(test_data):
        s = u'%s,%s\n' % (d[0], decode(d[1], result_avg[idx]))
        F.write(s)
    F.close()


# 拆分验证集
cv_path = os.path.join(model_cv_path, 'cv_0.pkl')
if not os.path.exists(cv_path):
    kf = KFold(n_splits=flodnums, shuffle=True,
               random_state=520).split(train_data)
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
    break
    print("kFlod ", i, "/", flodnums)
    # train_ = [train_data[i] for i in train_fold]
    # dev_ = [train_data[i] for i in test_fold]

    model = modify_bert_model_biMyGRU_crf()

    # train_D = data_generator(train_)
    # dev_D = data_generator(dev_)

    model_path = os.path.join(
        model_save_path, "modify_bert_biMyGRU_crf_model" + str(i) + ".weights")
    # if not os.path.exists(model_path):
    #     tbCallBack = TensorBoard(log_dir=os.path.join(model_save_path, 'logs_' + str(i)),  # log 目录
    #                              histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
    #                              batch_size=bsize,  # 用多大量的数据计算直方图
    #                              write_graph=True,  # 是否存储网络结构图
    #                              write_grads=False,  # 是否可视化梯度直方图
    #                              write_images=True,  # 是否可视化参数
    #                              embeddings_freq=0,
    #                              embeddings_layer_names=None,
    #                              embeddings_metadata=None)
    #
    #     evaluator = Evaluate(dev_, model_path)
    #     H = model.fit_generator(train_D.__iter__(),
    #                             steps_per_epoch=len(train_D),
    #                             epochs=15,
    #                             callbacks=[evaluator, tbCallBack],
    #                             validation_data=dev_D.__iter__(),
    #                             validation_steps=len(dev_D)
    #                             )
    #     # f = open(model_path.replace('.weights', 'history.pkl'), 'wb')
    #     # pickle.dump(H, f, 4)
    #     # f.close()
    #
    # print("load best model weights ...")
    # model.load_weights(model_path)
    #
    # print('val')
    # score.append(evaluate(dev_, batch=bsize * 2))
    # print("valid evluation:", score[-1])
    # print("valid score:", score)
    # print("valid mean score:", np.mean(score, axis=0))
    print('test')
    result_path = os.path.join(model_save_path, "train_ner_result_k" + str(i) + ".txt")
    test(test_data, result_path, batch=bsize)

    gc.collect()
    del model
    gc.collect()
    K.clear_session()
    a = 0 / 0

# %load_ext tensorboard  #使用tensorboard 扩展
# %tensorboard --logdir logs  #定位tensorboard读取的文件目录


# #  集成答案
# result = []
# for i, (train_fold, test_fold) in enumerate(kf):
#     print("kFlod ", i, "/", flodnums)
#     model = modify_bert_model_biMyGRU_crf()
#     model_path = os.path.join(
#         model_save_path, "modify_bert_biMyGRU_crf_model" + str(i) + ".weights")
#     print("load best model weights ...")
#     model.load_weights(model_path)
#     resulti = test_cv(test_data, batch=bsize)
#     result.append(resulti)
#     gc.collect()
#     del model
#     gc.collect()
#     K.clear_session()
#
# result_path = os.path.join(model_save_path, "train_ner_result_k" + 'cv' + ".txt")
# test_cv_decode(test_data, result, result_path)  # todo 优化

# 合并原始数据与预测的ner
'''
合并策略：
1、仅保留原始数据的ner，保留了A=NAN的数据（目前的）  67002
2、仅保留原始数据的ner，不保留A=NAN的数据（试过，效果比1差）  40284
3、保留原始数据的ner，对A=NAN的数据进行添加ner 正在尝试 72640
4、使用预测的ner修正原始的ner并人工加上classification标签
    4.1 仅修正A!=NAN的数据的标签，不保留A=NAN的数据（学弟） 67119  准确率0.52 召回率0.81
'''

# 策略3
data1path = train_data_path
data2path = os.path.join(model_save_path, "train_ner_result_k" + 'cv' + ".txt")
save_path = data2path.replace('.txt', '_strategy3.txt')
data1 = pd.read_csv(data1path, encoding='utf-8', sep=sep,
                    names=['id', 'text', 'Q', 'A'], quoting=csv.QUOTE_NONE)
data1.fillna('NaN', inplace=True)
data2 = pd.read_csv(data2path, encoding='utf-8', sep=',',
                    names=['id', 'A'], quoting=csv.QUOTE_NONE)
data2.fillna('NaN', inplace=True)
# todo 将A拆开
data21 = data2['A'].str.split(';', expand=True).stack()
data21 = data21.reset_index(level=1, drop=True).rename('A')
data2.drop('A', axis=1, inplace=True)
data2 = data2.join(data21)
data_nan = data1[data1.A == 'NaN']
data_nan.drop("A", axis=1, inplace=True)
data_nan = pd.merge(data_nan, data2, how='left', on=['id'])
data1 = data1[data1.A != 'NaN']
data = pd.concat([data1, data_nan], axis=0, ignore_index=True)
data.to_csv(save_path, sep=sep, columns=['id', 'text', 'Q', 'A'], header=False, index=False, encoding='utf-8')
print(data[data.Q == 'NaN'])
