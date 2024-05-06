import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from propy.PseudoAAC import GetAPseudoAAC
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import random






np.random.seed(101)

def GetSourceData(root, dir, lb):
    seqs = []
    print('\n')
    print('now is ', dir)
    file = '{}CD.txt'.format(dir)
    file_path = os.path.join(root, dir, file)

    with open(file_path) as f:
        for each in f:
            if each == '\n' or each[0] == '>':
                continue
            else:
                seqs.append(each.rstrip())

    # data and label
    label = len(seqs) * [lb]
    seqs_train, seqs_test, label_train, label_test = train_test_split(seqs, label, test_size=0.2, random_state=101)
    print('train data:', len(seqs_train))
    print('test data:', len(seqs_test))
    print('train label:', len(label_train))
    print('test_label:', len(label_test))
    print('total numbel:', len(seqs_train)+len(seqs_test))

    return seqs_train, seqs_test, label_train, label_test


def DataClean(data):
    max_len = 0
    for i in range(len(data)):
        st = data[i]
        # get the maximum length of all the sequences
        if(len(st) > max_len): max_len = len(st)

    return data, max_len

def PadEncode(data, max_len):
    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e = []
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i]
        if len(st)>max_len:
            st = st[:max_len]
        for j in st:
            index = amino_acids.index(j)
            elemt.append(index)
        if length > max_len:
            del elemt[max_len:-1]
        if length < max_len:
            elemt += [0] * (max_len - length)

        data_e.append(elemt)

    return data_e

def catch(data, label):
    # preprocessing label and data
    l = len(data)
    chongfu = 0
    for i in range(l):
        ll = len(data)
        idx = []
        each = data[i]
        j = i + 1
        bo = False
        while j < ll:
            if (data[j] == each).all():
                label[i] += label[j]
                idx.append(j)
                bo = True
            j += 1
        t = [i] + idx
        if bo:
            #print(t)
            chongfu += 1
            #print(data[t[0]])
            #print(data[t[1]])
        data = np.delete(data, idx, axis=0)
        label = np.delete(label, idx, axis=0)

        if i == len(data)-1:
            break
    print('total number of the same data: ', chongfu)

    return data, label

def catch_index(data, label, index):
    # preprocessing label and data
    l = len(data)
    chongfu = 0
    for i in range(l):
        ll = len(data)
        idx = []
        each = data[i]
        j = i + 1
        bo = False
        while j < ll:
            if (data[j] == each).all():
                label[i] += label[j]
                idx.append(j)
                bo = True
            j += 1
        t = [i] + idx
        if bo:
            #print(t)
            chongfu += 1
            #print(data[t[0]])
            #print(data[t[1]])
        data = np.delete(data, idx, axis=0)
        label = np.delete(label, idx, axis=0)
        index = np.delete(index, idx, axis=0)

        if i == len(data)-1:
            break
    print('total number of the same data: ', chongfu)

    return index

def GetSequenceData(dirs, root):
    # getting training data and test data
    count, max_length = 0, 0
    tr_data, te_data, tr_label, te_label = [], [], [], []
    for dir in dirs:
        # 1.getting data from file
        tr_x, te_x, tr_y, te_y = GetSourceData(root, dir, count)
        count += 1

        # 2.getting the maximum length of all sequences
        tr_x, len_tr = DataClean(tr_x)
        te_x, len_te = DataClean(te_x)
        if len_tr > max_length: max_length = len_tr
        if len_te > max_length: max_length = len_te

        # 3.dataset
        tr_data += tr_x
        te_data += te_x
        tr_label += tr_y
        te_label += te_y

    max_length = 400
    print(max_length)
    # data coding and padding vector to the filling length
    traindata = PadEncode(tr_data, max_length)
    testdata = PadEncode(te_data, max_length)
    # train_index, train_inds = aaindex1(tr_data, standardize='zscore')
    # test_index, test_inds = aaindex1(te_data, standardize='zscore')
    # train_index, train_inds = ngram(tr_data, n=2)
    # test_index, test_inds = ngram(te_data, n=2)
    # train_index = Convert_Seq2CKSAAP(tr_data)
    # test_index = Convert_Seq2CKSAAP(te_data)



    # data type conversion
    train_data = np.array(traindata)
    test_data = np.array(testdata)
    train_label = np.array(tr_label)
    test_label = np.array(te_label)
    # train_index = np.array(train_index)
    # test_index = np.array(test_index)


    return [train_data, test_data, train_label, test_label]


def GetData(path):
    dirs = ['AMP', 'ACP', 'ADP', 'AHP', 'AIP'] # functional peptides

    # get sequence data
    sequence_data = GetSequenceData(dirs, path)

    return sequence_data

path = 'data' # data path

sequence_data = GetData(path)
# sequence data partitioning
tr_seq_data,te_seq_data,tr_seq_label,te_seq_label = \
    sequence_data[0],sequence_data[1],sequence_data[2],sequence_data[3]
# tr_index, te_index = sequence_data[4],sequence_data[5]

tr_data = tr_seq_data
tr_label = tr_seq_label
te_data = te_seq_data
te_label = te_seq_label


train = [tr_data, tr_label]



test = [te_data, te_label]


print(train)

def data_train():
    return train

def data_test():
    return test


