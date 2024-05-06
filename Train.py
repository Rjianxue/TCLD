import math
import os
import numpy as np
import tensorflow as tf
from test import test_main
import time
from pathlib import Path
from Model import TCLD, Newnet
from Newnet import model_base
from Bio.SeqUtils.ProtParam import ProteinAnalysis

peptide_type = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
             'AVP',
             'BBP', 'BIP',
             'CPP', 'DPPIP',
             'QSP', 'SBP', 'THP']

tf.random.set_seed(101)
np.random.seed(101)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)


def counters(y_train):
    # counting the number of each peptide
    counterx = np.zeros(len(peptide_type) + 1, dtype='int')
    for i in y_train:
        a = np.sum(i)
        a = int(a)
        counterx[a] += 1
    print(counterx)

def train_method(train, para, model_num, model_path, data_size):
    # Implementation of training method
    Path(model_path).mkdir(exist_ok=True)

    # data get
    X_train, X_train_f, y_train = train[0], train[1], train[2]

    index = np.arange(len(y_train))
    np.random.shuffle(index)
    X_train = X_train[index]
    X_train_f = X_train_f[index]
    y_train = y_train[index]


    counters(y_train)

    # train
    length = X_train.shape[1]
    length_f = X_train_f.shape[1]
    out_length = y_train.shape[1]

    t_data = time.localtime(time.time())
    with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
        f.write('data process finished: {}m {}d {}h {}m {}s\n'.format(t_data.tm_mon, t_data.tm_mday, t_data.tm_hour,
                                                                      t_data.tm_min, t_data.tm_sec))

    class_weights = {}

    sumx = len(X_train)

    # 定义回调

    for m in range(len(data_size)):
        g = 5 * math.pow(int((math.log((sumx / data_size[m]), 2))), 2)
        if g <= 0:
            g = 1
        x = {m: g}
        class_weights.update(x)
    # c = 11
    # for counter in range(c, c + 1):
    for counter in range(1, model_num + 1):
        model = Newnet(length, length_f, out_length, para)
        # model = model_base(length, out_length, para)
        model.fit(X_train, y_train, epochs=200, batch_size=192, verbose=2,
                    class_weight=class_weights)

        each_model = os.path.join(model_path, 'model' + "_dropout_" + str(counter) + '.h5')

        model.save(each_model)

        tt = time.localtime(time.time())
        with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
            f.write('count{}: {}m {}d {}h {}m {}s\n'.format(str(counter), tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min,
                                                            tt.tm_sec))


def getSequenceData(first_dir, file_name):
    # getting sequence data and label
    data, label = [], []
    path = "{}/{}.txt".format(first_dir, file_name)

    with open(path) as f:
        for each in f:
            each = each.strip()
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
            else:
                data.append(each)

    return data, label

def PadEncode(data, label, max_len):  # 序列编码
    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, label_e = [], []
    sign = 0
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            index = amino_acids.index(j)
            elemt.append(index)
            sign = 0

        if length <= max_len and sign == 0:
            elemt += [0] * (max_len - length)
            data_e.append(elemt)
            label_e.append(label[i])
    return np.array(data_e), np.array(label_e)

def get_features(seq):
    # from PyBioMed.PyProtein import AAComposition
    # AAC = AAComposition.CalculateDipeptideComposition(seq)
    # r1 = list(AAC.values())
    # r1 = list(GetAPseudoAAC(seq,lamda=5).values())
    # r1 = AAIndex.GetAAIndex1(seq)
    x  = ProteinAnalysis(seq)
    r2 = [x.gravy()] #总平均亲水性
    r3 = [x.molecular_weight()] #分子量
    r4 = list(x.get_amino_acids_percent().values())  # 氨基酸百分比
    r5 = [x.charge_at_pH(pH=i) for i in range(14)]  # 在不同 pH 值下的电荷
    r6 = list(x.secondary_structure_fraction())   # 二级结构分数
    res = r2 + r3 + r4 + r5 + r6
    return res

def input_se(seqs_list):
    seqs = seqs_list
    elemt = []
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'

    for st in seqs:
        st = st.strip()
        invalid_amino_acid_found = False
        for j in st:
            if j not in amino_acids:
                invalid_amino_acid_found = True
                break
        if not invalid_amino_acid_found:
            elemt.append(st)

    lenth = len(elemt)
    seqx = []
    for i in range(lenth):
        seqs1 = get_features(elemt[i])
        seqx.append(seqs1)
    return np.array(seqx)

def staticTrainandTest(y_train, y_test):
    # static number
    data_size_tr = np.zeros(len(peptide_type))
    data_size_te = np.zeros(len(peptide_type))

    for i in range(len(y_train)):
        for j in range(len(y_train[i])):
            if y_train[i][j] > 0:
                data_size_tr[j] += 1

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] > 0:
                data_size_te[j] += 1

    print("TrainingSet:\n")
    for i in range(len(peptide_type)):
        print('{}:{}\n'.format(peptide_type[i], int(data_size_tr[i])))

    print("TestingSet:\n")
    for i in range(len(peptide_type)):
        print('{}:{}\n'.format(peptide_type[i], int(data_size_te[i])))

    return data_size_tr

def train_main(train, test, model_num, modelDir, data_size):
    # parameters
    ed = 128
    ps = 5
    fd = 128
    dp = 0.6
    lr = 0.001

    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}

    # Conduct training
    train_method(train, para, model_num, modelDir, data_size)

    # prediction
    test_main(test, para, model_num, modelDir)

    tt = time.localtime(time.time())
    with open(os.path.join(modelDir, 'time_demo.txt'), 'a+') as f:
        f.write(
            'test finish time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))


first_dir = 'dataset'

max_length = 50  # the longest length of the peptide sequence
# getting train data and test data
train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')
test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')

# Converting the list collection to an array
y_train = np.array(train_sequence_label)
y_test = np.array(test_sequence_label)

# The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
x_train, y_train = PadEncode(train_sequence_data, y_train, max_length)
x_test, y_test = PadEncode(test_sequence_data, y_test, max_length)
x_train_f = input_se(train_sequence_data)




# Counting the number of each peptide in the training set and the test set, and return the total number of the training set
data_size = staticTrainandTest(y_train, y_test)

x_train = x_train.astype(np.float32)
x_train_f = x_train_f.astype(np.float32)
y_train = y_train.astype(np.float32)

train = [x_train, y_train]
train_f = [x_train, x_train_f, y_train]
test = [x_test, y_test]
threshold = 0.5
model_num = 10  # model number
test.append(threshold)

modelDir = 'albtion_weight'

ed = 128
ps = 5
fd = 128
dp = 0.6
lr = 0.001

para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}

# Conduct training
train_method(train_f, para, model_num, modelDir, data_size)


