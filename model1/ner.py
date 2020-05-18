#! -*- coding: utf-8 -*-

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
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

maxlen = 256  # 140
learning_rate = 5e-5  # 5e-5
min_learning_rate = 1e-5  # 1e-5
bsize = 16
bert_kind = ['chinese_L-12_H-768_A-12', 'tf-bert_wwm_ext'][1]
config_path = os.path.join('../bert', bert_kind, 'bert_config.json')
checkpoint_path = os.path.join('../bert', bert_kind, 'bert_model.ckpt')
dict_path = os.path.join('../bert', bert_kind, 'vocab.txt')

# model_save_path = "./data/ccks2019_ckt/"
# train_data_path='./data/ccks2019/ccks2019_event_entity_extract/event_type_entity_extract_train.csv'
# test_data_path='./data/ccks2019/ccks2019_event_entity_extract/event_type_entity_extract_eval.csv'
# sep = ','

model_save_path = "./data/ccks2020_ckt_deleteTag/"
train_data_path = './data/event_entity_train_data_label.csv'
test_data_path = './data/classificationN_save/result_k0.csv'
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
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = maxlen
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


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

data['text'] = [delete_tag(s) for s in data.text]
# NaN替换成'NaN'
data.fillna('NaN', inplace=True)

data = data[data["Q"] != 'NaN']  # 优化 通过分类模型补上c
# 45796
classes = set(data["Q"].unique())

entity_train = list(set(data['A'].values.tolist()))

# ClearData
data.drop("id", axis=1, inplace=True)  # drop id
data.drop_duplicates(['text', 'Q', 'A'], keep='first', inplace=True)  # drop duplicates
data["A"] = data["A"].map(lambda x: str(x).replace('NaN', ''))
data["e"] = data.apply(lambda row: 1 if row[2] in row[0] else 0, axis=1)

data = data[data["e"] == 1]
# 38993
# D.drop_duplicates(["b", "c"], keep='first', inplace=True)  # drop duplicates
# 35288

train_data = []
for t, c, n in zip(data["text"], data["Q"], data["A"]):
    train_data.append((t, c, n))
print('最终训练集大小:%d' % len(train_data))
print('-' * 30)

D = pd.read_csv(test_data_path, header=None, sep=sep,
                names=["id", "text", "event"], quoting=csv.QUOTE_NONE)
D['text'] = [delete_tag(s) for s in D.text]
D.fillna('NaN', inplace=True)
# D['event'] = D['event'].map(lambda x: "公司股市异常" if x == "股市异常" else x)
D['text'] = D['text'].map(
    lambda x: x.replace("\x07", "").replace("\x05", "").replace("\x08", "").replace("\x06", "").replace("\x04", ""))
comp = re.compile(r"(\d{4}-\d{1,2}-\d{1,2})")
D['text'] = D['text'].map(lambda x: re.sub(comp, "▲", x))

test_data = []
for id, t, c in zip(D["id"], D["text"], D["event"]):
    test_data.append((id, t, c))
print('最终测试集大小:%d' % len(test_data))
print('-' * 30)

additional_chars = set()
for d in train_data:
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', d[2]))
if '，' in additional_chars:
    additional_chars.remove(u'，')


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
            X1, X2, S1, S2 = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text, c = d[0][:maxlen], d[1]
                text = u'___%s___%s' % (c, text)
                tokens = tokenizer.tokenize(text)
                e = d[2]
                e_tokens = tokenizer.tokenize(e)[1:-1]
                s1, s2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                start = list_find(tokens, e_tokens)
                if start != -1:
                    end = start + len(e_tokens) - 1
                    s1[start] = 1
                    s2[end] = 1
                    x1, x2 = tokenizer.encode(first=text)
                    X1.append(x1)
                    X2.append(x2)
                    S1.append(s1)
                    S2.append(s2)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    S1 = seq_padding(S1)
                    S2 = seq_padding(S2)
                    yield [X1, X2, S1, S2], None
                    X1, X2, S1, S2 = [], [], [], []


# 定义模型


def modify_bert_model_3():  # BiGRU + DNN #

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入
    s1_in = Input(shape=(None,))  # 实体左边界（标签）
    s2_in = Input(shape=(None,))  # 实体右边界（标签）

    x1, x2, s1, s2 = x1_in, x2_in, s1_in, s2_in
    # 标记text
    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)  # x_mask：1111111100
    x = bert_model([x1, x2])

    l = Lambda(lambda t: t[:, -1])(x)
    x = Add()([x, l])
    x = Dropout(0.1)(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

    x = Dense(1024, use_bias=False, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, use_bias=False, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(8, use_bias=False, activation='tanh')(x)

    ps1 = Dense(1, use_bias=False)(x)
    ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
    ps2 = Dense(1, use_bias=False)(x)
    ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

    model = Model([x1_in, x2_in], [ps1, ps2])

    train_model = Model([x1_in, x2_in, s1_in, s2_in], [ps1, ps2])

    loss1 = K.mean(K.categorical_crossentropy(s1_in, ps1, from_logits=True))
    ps2 -= (1 - K.cumsum(s1, 1)) * 1e10  # ？
    loss2 = K.mean(K.categorical_crossentropy(s2_in, ps2, from_logits=True))
    loss = loss1 + loss2

    train_model.add_loss(loss)
    train_model.compile(optimizer=Adam(learning_rate), metrics=['accuracy'])
    train_model.summary()
    return model, train_model


def modify_bert_model_h3():  # BiGRU + DNN #

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入
    s1_in = Input(shape=(None,))  # 实体左边界（标签）
    s2_in = Input(shape=(None,))  # 实体右边界（标签）

    x1, x2, s1, s2 = x1_in, x2_in, s1_in, s2_in
    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
    x = bert_model([x1, x2])

    l = Lambda(lambda t: t[:, -1])(x)
    x = Add()([x, l])
    x = Dropout(0.1)(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

    x = Dense(1024, use_bias=False, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, use_bias=False, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(8, use_bias=False, activation='tanh')(x)

    ps1 = Dense(1, use_bias=False)(x)
    ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
    ps2 = Dense(1, use_bias=False)(x)
    ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

    model = Model([x1_in, x2_in], [ps1, ps2])

    train_model = Model([x1_in, x2_in, s1_in, s2_in], [ps1, ps2])

    loss1 = K.mean(K.categorical_crossentropy(s1_in, ps1, from_logits=True))
    ps2 -= (1 - K.cumsum(s1, 1)) * 1e10
    loss2 = K.mean(K.categorical_crossentropy(s2_in, ps2, from_logits=True))
    loss = loss1 + loss2

    train_model.add_loss(loss)
    sgd = SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    train_model.compile(optimizer=sgd, metrics=['accuracy'])
    train_model.summary()
    return model, train_model


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


def extract_entity(text_in, c_in):
    """解码函数，应自行添加更多规则，保证解码出来的是一个公司名
    """
    if c_in not in classes:
        return 'NaN'
    text_in = u'___%s___%s' % (c_in, text_in)
    text_in = text_in[:maxlen - 2]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    _ps1, _ps2 = model.predict([_x1, _x2])
    _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
    for i, _t in enumerate(_tokens):
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            _ps1[i] -= 10
    start = _ps1.argmax()
    for end in range(start, len(_tokens)):
        _t = _tokens[end]
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            break
    end = _ps2[start:end + 1].argmax() + start
    a = text_in[start - 1: end]
    return a


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
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            print("save best model weights ...")
            train_model.save_weights(self.model_path)
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))

    def evaluate(self):
        A = 1e-10
        # F = open('dev_pred.json', 'w', encoding = 'utf-8')
        for d in tqdm(iter(self.dev_data)):
            R = extract_entity(d[0], d[1])
            if R == d[2]:
                A += 1
            # s = ', '.join(d + (R,))
            # F.write(s + "\n")
        # F.close()
        return A / len(self.dev_data)


def test(test_data, result_path):
    F = open(result_path, 'w', encoding='utf-8')
    for d in tqdm(iter(test_data)):
        s = u'%s\t%s\t%s\n' % (d[0], d[2], extract_entity(d[1], d[2]))
        # s = s.encode('utf-8')
        F.write(s)
    F.close()


def evaluate(dev_data):
    A = 1e-10
    # F = open('dev_pred.json', 'w', encoding='utf-8')
    for d in tqdm(iter(dev_data)):
        R = extract_entity(d[0], d[1])
        if R == d[2]:
            A += 1
    #     s = ', '.join(d + (R,))
    #     F.write(s + "\n")
    # F.close()
    return A / len(dev_data)


# Model
flodnums = 10

# 拆分验证集
cv_path = os.path.join(model_save_path, 'cv2.pkl')
if not os.path.exists(cv_path):
    y = [i[-2] for i in train_data]  # Q
    kf = StratifiedKFold(n_splits=flodnums, shuffle=True, random_state=520).split(train_data, y)
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

    model, train_model = modify_bert_model_3()

    train_D = data_generator(train_)
    dev_D = data_generator(dev_)

    model_path = os.path.join(model_save_path, "modify_bert_model" + str(i) + ".weights")
    if not os.path.exists(model_path):
        evaluator = Evaluate(dev_, model_path)
        train_model.fit_generator(train_D.__iter__(),
                                  steps_per_epoch=len(train_D),
                                  epochs=10,
                                  callbacks=[evaluator],
                                  validation_data=dev_D.__iter__(),
                                  validation_steps=len(dev_D)
                                  )
    else:
        print("load best model weights ...")
        train_model.load_weights(model_path)
        model.load_weights(model_path)
    del train_model
    gc.collect()
    del model
    gc.collect()
    K.clear_session()

    model, train_model = modify_bert_model_h3()

    model_h_path = os.path.join(model_save_path, "modify_bert_model_h" + str(i) + ".weights")
    if not os.path.exists(model_h_path):
        train_model.load_weights(model_path)
        evaluator = Evaluate(dev_, model_h_path)
        train_model.fit_generator(train_D.__iter__(),
                                  steps_per_epoch=len(train_D),
                                  epochs=15,
                                  callbacks=[evaluator],
                                  validation_data=dev_D.__iter__(),
                                  validation_steps=len(dev_D)
                                  )
    else:
        print("load best model weights ...")
        train_model.load_weights(model_h_path)
        model.load_weights(model_h_path)

    print('val')
    score.append(evaluate(dev_))
    print("valid evluation:", score[-1])
    print("valid score:", score)
    print("valid mean score:", np.mean(score))
    print('test')
    result_path = os.path.join(model_save_path, "result_k" + str(i) + ".txt")
    test(test_data, result_path)

    del train_model
    gc.collect()
    del model
    gc.collect()
    K.clear_session()
    a = 0 / 0

####### Submit #######
data = pd.DataFrame(columns=["sid", "tag", "company"])

dataid = pd.read_csv(os.path.join(model_save_path, "result_k0.txt"), sep=sep, names=["sid", "tag", "company"],
                     quoting=csv.QUOTE_NONE)[
    ['sid']]
dataid.fillna('NaN', inplace=True)

for i in range(1, flodnums):
    datak = pd.read_csv(os.path.join(model_save_path, "result_k" + str(i) + ".txt"), sep=sep,
                        names=["sid", "tag", "company"],
                        quoting=csv.QUOTE_NONE)
    datak.fillna('NaN', inplace=True)
    print(datak.shape)
    data = pd.concat([data, datak], axis=0)

submit = data.groupby(['sid', "tag", 'company'], as_index=False)['sid', "tag"].agg({"count": "count"})

print(submit.shape)
print(submit[submit.company == 'NaN'])

submit = submit.sort_values(by=["sid", "tag", "count"], ascending=False).groupby(["sid", "tag"], as_index=False).first()

print(submit.shape)

submit = dataid.merge(submit, how='left', on=['sid', "tag"]).fillna("NaN")
print(data[['sid', "tag"]].drop_duplicates().shape)  # ??
print(submit.shape)

submit[['sid', "tag", 'company']].to_csv(os.path.join(model_save_path, "result.txt"), header=None, index=False, sep=sep)

print(submit)
