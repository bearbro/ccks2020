'''
# 拆封次数
    # 数据拆分
    # 将验证集的实体 从训练集中去掉

    # 训练模型 （终止条件，保留的模型）todo 保留最后的 or 验证集上分数最高的？
    # 在验证集上预测

# 通过验证集上的预测统计各样本的错误次数

# 根据错误次数计算新权重

# 使用新权重训练模型（终止条件，保留的模型 ）todo 是否划分验证集？ 保留最后的 or 验证集上分数最高的？

# 在测试集上预测
'''

# 参数


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
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
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
from keras.callbacks import TensorBoard

split_nums = 3
flod_nums = 5
mistake_eps = 0.7
'''
mistake_eps=1-p
令p为实际检测到的标签错误数与检测到的标签错误数之比。通过从检测到的标签错误中手动检查随机样本，可以大致估算出p。
'''
max_epochs = 15
maxlen = 256  # 140
learning_rate = 5e-5  # 5e-5
min_learning_rate = 1e-5  # 1e-5
bsize = 16
bert_kind = ['chinese_L-12_H-768_A-12', 'tf-bert_wwm_ext'][0]
config_path = os.path.join('../bert', bert_kind, 'bert_config.json')
checkpoint_path = os.path.join('../bert', bert_kind, 'bert_model.ckpt')
dict_path = os.path.join('../bert', bert_kind, 'vocab.txt')

model_cv_path = "./data/ccks2020_ckt_256_crossweight_bilstm_crf/"
model_save_path = "./data/ccks2020_ckt_256_crossweight_bilstm_crf/"
train_data_path = '../model2/data/event_entity_train_data_label.csv'
test_data_path = '../model2/data/event_entity_dev_data.csv'
sep = '\t'

for p in [model_cv_path, model_save_path]:
    if not os.path.exists(p):
        os.mkdir(p)

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
data = data[data.A != 'NaN']  # todo

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

BIOtag = ['O', 'B', 'I']  # todo 可以改用BIOES
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
        padding_wd = [padding] * len(X[0][0])
        padding_wd[tag2id['O']] = 1
        return np.array([
            np.concatenate([x, [padding_wd] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])


class data_generator:
    def __init__(self, data, batch_size=bsize, weight=False):
        self.data = data
        self.batch_size = batch_size
        self.weight = weight
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, P, W = [], [], [], []
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
                if self.weight:
                    W.append(d[2])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    P = seq_padding(P, wd=2)
                    # print(X1.shape)
                    # print(P.shape)
                    if self.weight:
                        W = np.array(W)
                        yield [X1, X2], P, W
                    else:
                        yield [X1, X2], P
                    X1, X2, P, W = [], [], [], []


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
    #  [batch,maxL,768] -》[batch,maxL,3]
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
    outputs = Dense(units=len(BIOtag), use_bias=False, activation='Softmax')(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_masking_model')
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
    # # Masking
    # xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
    # outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
    # outputs = Masking(mask_value=0)(outputs)
    outputs = Bidirectional(LSTM(units=300, return_sequences=True))(outputs)
    outputs = Dropout(0.2)(outputs)
    crf = CRF(len(BIOtag), sparse_target=False)
    outputs = crf(outputs)

    model = keras.models.Model([x1_in, x2_in], outputs, name='basic_bilstm_crf_model')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    return model


def modify_bert_model_3_crf():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function, metrics=[crf.accuracy])
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
    r = list(set(r))
    r.sort()
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
        s = u'%s,%s\n' % (d[0], extract_entity(d[1]))
        F.write(s)
    F.close()


def test_cv(test_data):
    '''预测'''
    r = []
    for d in tqdm(iter(test_data)):
        text_in = d[1][:maxlen]
        _tokens = tokenizer.tokenize(text_in)
        _x1, _x2 = tokenizer.encode(text_in)
        _x1, _x2 = np.array([_x1]), np.array([_x2])
        _p = model.predict([_x1, _x2])[0]
        r.append(_p)
    return r


def test_cv_decode(test_data, result, result_path):
    '''获得cv的结果'''
    result_avg = np.mean(result, axis=0)
    F = open(result_path, 'w', encoding='utf-8')
    for idx, d in enumerate(test_data):
        s = u'%s,%s\n' % (d[0], decode(d[1], result_avg[idx]))
        F.write(s)
    F.close()


def evaluate_cw(dev_data, val_result_path):
    A = 1e-10
    F = 1e-10
    val_result = []
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
        val_result.append(y)
    # 持久化
    f = open(val_result_path, 'wb')
    pickle.dump(val_result, f, 4)
    f.close()

    return A / len(dev_data), F / len(dev_data)


# 将dev中出现过的实体从train中删除
def delete_tag_in_dev(train_, dev_):
    # train text,A[,W]
    A_set = set()
    for i in dev_:
        A_set.update(i[1].split(';'))
    new_train_ = []
    for j in train_:
        jset = set(j[1].split(';'))
        if jset & A_set == set():
            new_train_.append(j)
    return new_train_


def RandomGroupKFold_split(groups, n, seed=None):  # noqa: N802
    """
    Random analogous of sklearn.model_selection.GroupKFold.split.

    :return: list of (train, test) indices
    """
    groups = pd.Series(groups)
    ix = np.arange(len(groups))
    unique = np.unique(groups)
    np.random.RandomState(seed).shuffle(unique)
    result = []
    for split in np.array_split(unique, n):
        mask = groups.isin(split)
        train, test = ix[~mask], ix[mask]
        result.append((train, test))

    return result


# Model
flodnums = flod_nums
for split_idx in range(split_nums):
    # 拆分验证集
    cv_path = os.path.join(model_cv_path, 'cv_%d.pkl' % split_idx)
    if not os.path.exists(cv_path):
        kf = KFold(n_splits=flodnums, shuffle=True, random_state=split_idx).split(train_data)
        # A_list = [i[1].split(';')[0] for i in train_data]
        # kf = RandomGroupKFold_split(A_list, flodnums, seed=split_idx)  # todo 按组划分
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
        # break
        print(('%d/%d' % (split_idx, split_nums)), "kFlod ", i, "/", flodnums)
        train_ = [train_data[i] for i in train_fold]
        dev_ = [train_data[i] for i in test_fold]
        # 将dev中出现过的实体从train中删除
        # train_ = delete_tag_in_dev(train_, dev_)

        model = modify_bert_model_bilstm_crf()

        train_D = data_generator(train_)
        dev_D = data_generator(dev_)

        model_path = os.path.join(model_save_path,
                                  str(split_idx) + '_' + "modify_bert_bilstm_crf_model" + str(i) + ".weights")
        if not os.path.exists(model_path):
            tbCallBack = TensorBoard(log_dir=os.path.join(model_save_path, str(split_idx) + '_' + 'logs_' + str(i)),
                                     # log 目录
                                     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                     batch_size=bsize,  # 用多大量的数据计算直方图
                                     write_graph=True,  # 是否存储网络结构图
                                     write_grads=False,  # 是否可视化梯度直方图
                                     write_images=True,  # 是否可视化参数
                                     embeddings_freq=0,
                                     embeddings_layer_names=None,
                                     embeddings_metadata=None)
            evaluator = Evaluate(dev_, model_path)
            model.fit_generator(train_D.__iter__(),
                                steps_per_epoch=len(train_D),
                                epochs=max_epochs,
                                callbacks=[evaluator, tbCallBack],
                                validation_data=dev_D.__iter__(),
                                validation_steps=len(dev_D)
                                )

        print("load best model weights ...")
        model.load_weights(model_path)
        val_result_path = os.path.join(model_save_path, str(split_idx) + '_' + "val_result_k" + str(i) + ".pkl")
        print('val')
        score.append(evaluate_cw(dev_, val_result_path))
        print("valid evluation:", score[-1])
        print("valid score:", score)
        print("valid mean score:", np.mean(score, axis=0))
        # print('test')
        # result_path = os.path.join(model_save_path, str(split_idx)+'_'+"result_k" + str(i) + ".txt")
        # test(test_data, result_path)

        gc.collect()
        del model
        gc.collect()
        K.clear_session()

# 通过验证集上的预测统计各样本的错误次数

mistake_count = dict()
for split_idx in range(split_nums):
    cv_path = os.path.join(model_cv_path, 'cv_%d.pkl' % split_idx)
    assert os.path.exists(cv_path)
    f = open(cv_path, 'rb')
    kf = pickle.load(f)
    f.close()

    for i, (train_fold, test_fold) in enumerate(kf):
        print(('%d/%d' % (split_idx, split_nums)), "kFlod ", i, "/", flodnums)
        dev_ = [train_data[i] for i in test_fold]
        val_result_path = os.path.join(model_save_path, str(split_idx) + '_' + "val_result_k" + str(i) + ".pkl")
        assert os.path.exists(val_result_path)
        f = open(val_result_path, 'rb')
        val_result = pickle.load(f)
        f.close()
        for dev_i in range(len(dev_)):
            Y = dev_[dev_i][1]  # ner1;ner2
            y = val_result[dev_i]  # ner1;ner2
            if set(Y.split(';')) != set(y.split(';')):
                mistake_count[dev_[dev_i][0]] = mistake_count.get(dev_[dev_i][0], 0) + 1

# 根据错误次数计算新权重
train_data_w = []
for i in train_data:
    t = i[0]
    a = i[1]
    w = mistake_eps ** mistake_count.get(t, 0)
    train_data_w.append((t, a, w))

# 使用新权重训练模型（终止条件，保留的模型 ）
# 在测试集上预测


flodnums = flod_nums

# 拆分验证集
cv_path = os.path.join(model_cv_path, 'cv_final.pkl')
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
    # break
    print('final', "kFlod ", i, "/", flodnums)
    train_ = [train_data_w[i] for i in train_fold]
    dev_ = [train_data_w[i] for i in test_fold]
    # 将dev中出现过的实体从train中删除
    # train_ = delete_tag_in_dev(train_, dev_)

    model = modify_bert_model_bilstm_crf()

    train_D = data_generator(train_, weight=True)
    dev_D = data_generator(dev_, weight=True)

    model_path = os.path.join(model_save_path,
                              str(split_idx) + '_' + "modify_bert_bilstm_crf_model" + str(i) + ".weights")
    if not os.path.exists(model_path):
        tbCallBack = TensorBoard(log_dir=os.path.join(model_save_path, 'finall_' + 'logs_' + str(i)),
                                 # log 目录
                                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                 batch_size=bsize,  # 用多大量的数据计算直方图
                                 write_graph=True,  # 是否存储网络结构图
                                 write_grads=False,  # 是否可视化梯度直方图
                                 write_images=True,  # 是否可视化参数
                                 embeddings_freq=0,
                                 embeddings_layer_names=None,
                                 embeddings_metadata=None)
        evaluator = Evaluate(dev_, model_path)
        model.fit_generator(train_D.__iter__(),
                            steps_per_epoch=len(train_D),
                            epochs=max_epochs,
                            callbacks=[evaluator, tbCallBack],
                            validation_data=dev_D.__iter__(),
                            validation_steps=len(dev_D)
                            )

    print("load best model weights ...")
    model.load_weights(model_path)
    val_result_path = os.path.join(model_save_path, str(split_idx) + '_' + "val_result_k" + str(i) + ".pkl")
    print('val')
    score.append(evaluate_cw(dev_, val_result_path))
    print("valid evluation:", score[-1])
    print("valid score:", score)
    print("valid mean score:", np.mean(score, axis=0))
    # print('test')
    # result_path = os.path.join(model_save_path, str(split_idx)+'_'+"result_k" + str(i) + ".txt")
    # test(test_data, result_path)

    gc.collect()
    del model
    gc.collect()
    K.clear_session()
    break

'''
预计耗时
1个model 4~5h
3*5+1个模型
1天4个model，要4天
'''
