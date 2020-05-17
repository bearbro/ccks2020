import csv

import pandas as pd

data_path = './data/event_entity_train_data_label.csv'
data_test_path = './data/event_entity_dev_data.csv'
sep = '\t'
data = pd.read_csv(data_path, encoding='utf-8', sep=sep, index_col=None, header=None,
                   names=['id', 'text', 'Q', 'A'], quoting=csv.QUOTE_NONE)
data.fillna('NaN', inplace=True)
id2label = sorted(list(set(data.Q)))  # 里面有NaN
id2label.remove('NaN')
label2id = {v: i for i, v in enumerate(id2label)}

classify_path = './data/classificationN_save/result_k0.csv'
classify_path_new = './data/classificationN_save/result_k0add.csv'


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


with open(classify_path, 'r') as fr:
    with open(classify_path_new, 'w') as fout:
        lines = fr.readlines()
        lset = set()
        for line in lines:
            line = line[:-1]
            id, text, c = line.split('\t')
            if c == 'NaN':
                tag = []
            else:
                tag = [c]
            for i in id2label:
                if have(text, i) and i not in tag:
                    tag.append(i)
            for i in tag:
                l = '%s\t%s\t%s\n' % (id, text, i)
                if l not in lset:
                    fout.write(l)
                    lset.add(l)
            if len(tag) == 0:
                fout.write('%s\t%s\t%s\n' % (id, text, c))
