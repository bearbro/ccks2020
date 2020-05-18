# 文本  多标签分类
import gc
import pickle

import pandas as pd
import csv, os
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import keras
import codecs

from tqdm import tqdm


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

data_path = './data/event_entity_train_data_label.csv'
data_test_path = './data/event_entity_dev_data.csv'
sep = '\t'


class Config:
    bert_path = '../bert/chinese_L-12_H-768_A-12'
    bert_config_path = os.path.join(bert_path, 'bert_config.json')
    bert_ckpt_path = os.path.join(bert_path, 'bert_model.ckpt')
    bert_dict_path = os.path.join(bert_path, 'vocab.txt')
    ckpt_path = './data/classificationN_ckpt'
    save_path = './data/classificationN_save'
    max_length = 128
    batch_size = 32
    learning_rate = 1e-5


config = Config()
for pathi in [config.ckpt_path, config.save_path]:
    if not os.path.exists(pathi):
        os.mkdir(pathi)

# # 2019
# data_path = './data/ccks2019/ccks2019_event_entity_extract/event_type_entity_extract_train.csv'
# data_test_path = './data/ccks2019/ccks2019_event_entity_extract/event_type_entity_extract_eval.csv'
# sep = ','

data = pd.read_csv(data_path, encoding='utf-8', sep=sep, index_col=None, header=None,
                   names=['id', 'text', 'Q', 'A'], quoting=csv.QUOTE_NONE)

# 所有id都不同，共72515行
print('原始数据有%d行' % len(data))
# 存在仅id不同的行,
# 去除text,Q,A相同的重复行
data.drop_duplicates(subset=['text', 'Q', 'A'], keep='first', inplace=True)
data.index = range(len(data))
print('去除text,Q,A重复的行后，还有%d行' % len(data))
# NaN替换成'NaN'
data.fillna('NaN', inplace=True)
print('共%d种text' % len(set(data.text)))

# 去除A列
data.drop(labels='A', axis=1, inplace=True)
data.drop(labels='id', axis=1, inplace=True)

# 改成二分类的数据集 text，Q，label
id2label = sorted(list(set(data.Q)))  # 里面有NaN
id2label.remove('NaN')
label2id = {v: i for i, v in enumerate(id2label)}
print('共%d种Q' % len(id2label))

# 构造多标签数据集
print('处理后数据个数%d' % len(data))
w = data.groupby(['text'])['Q'].count().reset_index()
data_new = data.drop_duplicates(subset=['text', 'Q'], keep='first')
data_new = data_new.groupby(['text'])['Q'].apply(lambda x: [i for i in x]).reset_index()


def f(tags):
    r = [0] * len(id2label)
    for i in tags:
        if i in label2id:
            r[label2id[i]] = 1
    return r


data_new['Q'] = [f(i) for i in data_new.Q]
data = data_new
'''
data【text，label】

'''

# 测试集
test_data = pd.read_csv(data_test_path, encoding='utf-8', sep=sep, index_col=None, header=None,
                        names=['id', 'text'], quoting=csv.QUOTE_NONE)
test_data = test_data.values
# 作用
'''
本来 Tokenizer 有自己的 _tokenize 方法，我这里重写了这个方法，是要保证 tokenize 之后的结果，跟原来的字符串长度等长（如果算上两个标记，那么就是等长再加 2）。 Tokenizer 自带的 _tokenize 会自动去掉空格，然后有些字符会粘在一块输出，导致 tokenize 之后的列表不等于原来字符串的长度了，这样如果做序列标注的任务会很麻烦。

而为了避免这种麻烦，还是自己重写一遍好了。主要就是用 [unused1] 来表示空格类字符，而其余的不在列表的字符用 [UNK] 表示，其中 [unused*] 这些标记是未经训练的（随即初始化），是 Bert 预留出来用来增量添加词汇的标记，所以我们可以用它们来指代任何新字符。
'''
token_dict = {}

with codecs.open(config.bert_dict_path, 'r', 'utf8') as reader:
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


# 让每条文本的长度相同，用0填充
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = config.max_length
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


# 计算：F1值
def f1_metric(y_true, y_pred):
    '''
    metric from here
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''

    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# 构建模型 todo
def basic_network():
    bert_model = load_trained_model_from_checkpoint(config.bert_config_path,
                                                    config.bert_ckpt_path,
                                                    seq_len=config.max_length,
                                                    training=False,
                                                    trainable=True)

    x_1 = keras.layers.Input(shape=(config.max_length,), name='input_x1')
    x_2 = keras.layers.Input(shape=(config.max_length,), name='input_x2')

    bert_out = bert_model([x_1, x_2])  # 输出维度为(batch_size,max_length,768)

    # dense=bert_model.get_layer('NSP-Dense')
    bert_out1 = keras.layers.Lambda(lambda bert_out: bert_out[:, 0], name='bert_1')(bert_out)
    bert_out_next = bert_out1
    outputs = keras.layers.Dense(len(id2label), activation='sigmoid', name='dense')(bert_out_next)

    model = keras.models.Model([x_1, x_2], outputs, name='basic_model')
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', f1_metric]
    )
    model.summary()
    return model


class MyLayer(Layer):
    """输入bert最后一层的embedding和位置信息token_ids

    在这一层将embedding的第一位即cls和句子B的embedding的平均值拼接

    # Arguments
        result: 输出的矩阵纬度（batchsize,output_dim).
    """

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):  # 2*(batch_size,max_length,768)
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        # no need
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        bert_out, x1 = x
        pooled_output = bert_out[:, 0]  # （batch_size，hidden_​​size）
        target = tf.multiply(bert_out, K.expand_dims(x1, -1))  # 提取句子B（公司实体）的sequence_output  expand_dims和unsqueeze
        # sequence方向上求和,形状为（batch_size，sequenceB_length，hidden_​​size）-》（batch_size，hidden_​​size）tf.div
        target = K.sum(target, axis=1)
        target_div = K.sum(x1, axis=1)  # 得到句子B的长度
        target = tf.div(target, K.expand_dims(target_div, -1))  # 获得平均数，现状（batch_size，hidden_​​size） tf.div divide
        target_cls = K.concatenate([target, pooled_output], axis=-1)  # 拼接

        return target_cls

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return (shape_a[0], shape_a[-1] * 2)  # (batch_size,768*2)


def cls_context_network():
    bert_model = load_trained_model_from_checkpoint(config.bert_config_path,
                                                    config.bert_ckpt_path,
                                                    seq_len=config.max_length,
                                                    training=False,
                                                    trainable=True,
                                                    output_layer_num=1  # 选几层
                                                    )
    # 选择某些层进行训练
    # bert_model.summary()
    # for l in bert_model.layers:
    #     # print(l)
    #     l.trainable = True
    x1 = keras.layers.Input(shape=(config.max_length,))  # 位置000111000
    x2 = keras.layers.Input(shape=(config.max_length,))  # 字id
    bert_out = bert_model([x1, x2])  # 输出维度为(batch_size,max_length,768)
    print(bert_out.shape)
    # dense=bert_model.get_layer('NSP-Dense')
    bert_out = keras.layers.Lambda(lambda bert_out: bert_out)(bert_out)
    bert_out = MyLayer()([bert_out, x1])
    # bert_out = keras.layers.Lambda(lambda bert_out: bert_out[:, 0])(bert_out)
    # bert_out=keras.layers.Dropout(0.2)(bert_out)
    outputs = keras.layers.Dense(1, activation='sigmoid')(bert_out)

    model = keras.models.Model([x1, x2], outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', f1_score]
    )
    model.summary()
    return model


class data_generator:
    def __init__(self, data, batch_size=config.batch_size):
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
                text, label = d[0], d[1]
                # 构造位置id和字id
                token_ids, segment_ids = tokenizer.encode(first=text, max_len=config.max_length)
                X1.append(token_ids)
                X2.append(segment_ids)
                P.append(label)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    # X1 = seq_padding(X1)
                    # X2 = seq_padding(X2)
                    X1 = np.array(X1)
                    X2 = np.array(X2)
                    P = np.array(P)
                    yield [X1, X2], P
                    X1, X2, P = [], [], []


def have(text, c):
    dict_c = {
        '债务违约': ['债务违约', '货款违约'],
        '履行连带担保责任': ['履行连带担保责任', '承担连带担保责任']
    }
    if c in dict_c:
        c = dict_c[c]
    else:
        c = [c]
    for i in c:
        if i in text:
            return True
    return False


def test(text_in, result_path, add=False):
    with open(result_path, 'w') as fout:
        for d in tqdm(iter(text_in)):
            id = d[0]
            text_in = d[1]
            # 构造位置id和字id
            token_ids, segment_ids = tokenizer.encode(first=text_in, max_len=config.max_length)
            p = train_model.predict([[token_ids], [segment_ids]])[0]
            tags = [id2label[i] for i, v in enumerate(p) if v > 0.5]
            if add:
                # 显式出现的词
                for idx, tag in enumerate(id2label):
                    if tag not in tags and have(text_in, tag):  # 硬规则 可优化
                        tags.append(tag)
            if len(tags) == 0:
                fout.write('%d\t%s\t%s\n' % (id, text_in, 'NaN'))
            else:
                for tag in tags:
                    fout.write('%d\t%s\t%s\n' % (id, text_in, tag))


def extract_entity(text_in, add=False):
    token_ids, segment_ids = tokenizer.encode(first=text_in, max_len=config.max_length)
    p = train_model.predict([[token_ids], [segment_ids]])[0]
    if add:
        # 显式出现的词
        for idx, tag in enumerate(id2label):
            if have(text_in, tag):  # 硬规则 可优化
                p[idx] = 1
    return [1 if i > 0.5 else 0 for i in p]


def evaluate(dev_data, add=False):
    A = 1e-10
    F = 1e-10
    true_m = []
    pred_m = []
    for d in tqdm(iter(dev_data)):
        R = extract_entity(d[0], add=add)
        if R == d[1]:
            A += 1
        true_m.append(d[1])
        pred_m.append(R)
    return A / len(dev_data), f1_score(true_m, pred_m, average='micro')


# 拆分验证集
flodnums = 5
cv_path = os.path.join(config.ckpt_path, 'cv.pkl')
train_data = list(data.text)
if not os.path.exists(cv_path):
    y = []
    for i in data.Q:
        yi = sum([(2 ** idx) * v for idx, v in enumerate(i)])
        y.append(yi)
        # if 1 in i:
        #     y.append(i.index(1))
        # else:
        #     y.append(-1)
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
score_add = []
for i, (train_fold, test_fold) in enumerate(kf):
    print("kFlod ", i, "/", flodnums)
    train_ = data[data.text.isin([train_data[i] for i in train_fold])].values
    dev_ = data[data.text.isin([train_data[i] for i in test_fold])].values

    train_D = data_generator(train_)
    dev_D = data_generator(dev_)

    train_model = basic_network()
    model_path = os.path.join(config.ckpt_path, "modify_bert_model-" + str(i) + ".weights")
    if not os.path.exists(model_path):
        checkpoint = keras.callbacks.ModelCheckpoint(model_path,
                                                     monitor='val_f1_metric',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max',
                                                     save_weights_only=True,
                                                     period=1)

        earlystop = keras.callbacks.EarlyStopping(monitor='val_f1_metric',
                                                  patience=3,
                                                  verbose=0,
                                                  mode='max')

        train_model.fit_generator(train_D.__iter__(),
                                  steps_per_epoch=len(train_D),
                                  epochs=10,
                                  callbacks=[checkpoint, earlystop],
                                  validation_data=dev_D.__iter__(),
                                  validation_steps=len(dev_D)
                                  )
    else:
        train_model.load_weights(model_path)
    print('val')
    score.append(evaluate(dev_))
    print("val evluation", score[-1])
    print("valid score:", score)
    print("valid mean score:", np.mean(score, axis=0))
    score_add.append(evaluate(dev_, add=True))
    print("val evluation_add", score_add[-1])
    print("val score_add:", score_add)
    print("val mean score_add:", np.mean(score_add, axis=0))
    result_path = os.path.join(config.save_path, "result_k" + str(i) + ".csv")
    if not os.path.exists(result_path):
        print('test')
        test(test_data, result_path, add=False)

    del train_model
    gc.collect()
    K.clear_session()
    break

# todo 集成答案
