import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
import seaborn as sns

data_path = './data/event_entity_train_data_label.csv'
data_test_path = './data/event_entity_dev_data.csv'
sep = '\t'

# # 2019
# data_path = './data/ccks2019/ccks2019_event_entity_extract/event_type_entity_extract_train.csv'
# data_test_path = './data/ccks2019/ccks2019_event_entity_extract/event_type_entity_extract_eval.csv'
# sep = ','

data = pd.read_csv(data_path, encoding='utf-8', sep=sep, index_col=None, header=None,
                   names=['id', 'text', 'Q', 'A'], quoting=csv.QUOTE_NONE)

# 确认数据个数
# 每行都是3个\t,即4段
with open('./data/event_entity_train_data_label.csv', 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    lines[0] = lines[0].replace('\ufeff', '')
    text_set = dict()
    for i in lines:
        i_size = len(i.split('\t'))
        if i_size != 4:
            print('error', i)
        key = i.split('\t')[0]
        if key in text_set:
            text_set[key] = text_set[key] + 1
        else:
            text_set[key] = 1
    text_set = {k: n for k, n in text_set.items() if n > 1}
    for k, n in text_set.items():
        print(n, k)
    miss = set(text_set.keys()) - set(
        data.id.to_list())
    miss = list(miss)
    print(len(miss))  # 缺少的数据

# 所有id都不同，共72515行
print('原始数据有%d行' % len(data))
# 存在仅id不同的行,
# 去除仅id不同的重复行,剩余67002行
data.drop_duplicates(subset=['text', 'Q', 'A'], keep='first', inplace=True)
data.index = range(len(data))
print('去除text,Q,A重复的行后，还有%d行' % len(data))
# text有重复 60242种
# 看一下text的长度
TL = [len(i) for i in data.text]
sns.distplot(TL)
plt.show()
print('text-min:%d' % min(TL))
print('text-max:%d' % max(TL))
print('text-[,10):%d\t%.3f%%' % (len([i for i in TL if i < 10]), len([i for i in TL if i < 10]) / len(TL) * 100))
print('text-(256,):%d\t%.3f%%' % (len([i for i in TL if i > 256]), len([i for i in TL if i > 256]) / len(TL) * 100))
print('text-(512,):%d\t%.3f%%' % (len([i for i in TL if i > 512]), len([i for i in TL if i > 512]) / len(TL) * 100))
# 最小3 最大 2646
# 小于10 104
# 大于512 449
# 大于256 1917


# text+Q 重复的 2584种，6560行
text_set = dict()
for i in range(len(data)):
    key = data.text[i] + str(data.Q[i])
    if key in text_set:
        text_set[key] = text_set[key] + [data.A[i]]
    else:
        text_set[key] = [data.A[i]]
text_set = {k: a for k, a in text_set.items() if len(a) > 1}
print('text+Q 重复的 %d种' % len(text_set))  # 2584种
print('text+Q 重复的 %d行' % sum([len(i) for i in text_set.values()]))  # 6560个
# 画图看一眼
v = [len(i) for i in text_set.values()]
x = sorted(list(set(v)))
y = [v.count(i) for i in x]
# import matplotlib.pyplot as plt
# plt.bar(x,y)
# plt.show()
# plt.scatter(x,y)
# plt.show()
# A的个数即出现频率
# [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 19, 163]
# [1945, 391, 123, 59, 15, 10, 19, 2, 12, 4, 1, 2, 1]

# 推测 text+Q 相同 A不同时，A连在一起 ， todo 验证 错的，仅部分连在一起
tq2a = {k: a for k, a in text_set.items() if len(a) > 1}

global a_sly, a_sly_h, a_sly_1
# 答案中缩略语个数
a_sly = 0
# 存在缩略语的TQ个数
a_sly_h = 0
# 答案中缩略语个数
a_sly_1 = 0


def AisNeighbour(TQ, A):
    """答案是否连续出现"""

    def add_gang(s):
        """避免正则查找的时候出错，进行转义"""

        def add(i):
            return '\\' + i

        tszf = '\\(){}[]?+*.'  # 需要添加/的符号组
        for k in tszf:
            s = s.replace(k, add(k))
        return s

    # 答案中有同时存在简称和全称的情况 例如'昌达胜科网络公司','北京昌达胜科网络公司'
    # 去除简称
    delA = []
    for i in A:
        for j in A:
            if i != j and i in j:
                delA.append(i)
    global a_sly, a_sly_h, a_sly_1
    a_sly += len(delA)
    if len(delA):
        a_sly_h += 1
    if len(delA) == 1:
        a_sly_1 += 1
    A = [i for i in A if i not in delA]
    # 避免正则查找的时候出错，进行转义
    A = [add_gang(i) for i in A]
    # 消除答案顺序的影响
    # 对A全排序
    from itertools import permutations
    for Ai in permutations(A):
        rs = '(.{0,2}|["|”|]{0,1}.{1,2}["|“]{0,1}|[（|(| (].{0,2}[简]{0,1}称.{1})'.join(Ai)
        # print(A)
        matchObj = re.search(rs, TQ, re.M | re.I)
        if matchObj:
            return matchObj.group()
        if len(A) > 5:  # 11 20
            break
    return False


a_near_a = dict()
not_a_near_a = dict()
for k, a in tq2a.items():
    if len(a) > 1:
        near = AisNeighbour(k, a)
        if near:
            # a_near_a[str(len(k)) + '---' + AisNeighbour(k, a)] = a
            a_near_a[k + '---' + near] = a
        else:
            not_a_near_a[k] = a

print('答案贴近%d个\n答案不贴近%d个' % (len(a_near_a), len(not_a_near_a)))

# 存在缩略语的TQ个数
print('存在缩略语的TQ个数：%d' % a_sly_h)
print('A中缩略语个数：%d' % a_sly)
print('A中仅缩略语重复的TQ个数：%d' % a_sly_1)

# 测试集合
data_test = pd.read_csv(data_test_path, encoding='utf-8', sep=sep, index_col=None, header=None,
                        names=['id', 'text'], quoting=csv.QUOTE_NONE)
print('测试集合大小:%d' % len(data_test))  # 900
print('测试集合id种类:%d' % len(set(data_test.id)))  # 900
print('测试集合text种类:%d' % len(data_test.text))  # 900

TL_test = [len(i) for i in data_test.text]
sns.distplot(TL_test)
plt.show()
print('text-min:%d' % min(TL_test))
print('text-max:%d' % max(TL_test))
print('text-[,10):%d\t%.3f%%' % (
    len([i for i in TL_test if i < 10]), len([i for i in TL_test if i < 10]) / len(TL_test) * 100))
print('text-(256,):%d\t%.3f%%' % (
    len([i for i in TL_test if i > 256]), len([i for i in TL_test if i > 256]) / len(TL_test) * 100))
print('text-(512,):%d\t%.3f%%' % (
    len([i for i in TL_test if i > 512]), len([i for i in TL_test if i > 512]) / len(TL_test) * 100))
# 最小16,最大343
# 小于10 0
# 大于512 0
# 大于256 18
