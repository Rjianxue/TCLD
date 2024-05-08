import os
from evaluation import evaluate
from tensorflow.keras.models import load_model
from model_ import MultiHeadAttention, multilabel_categorical_crossentropy, TransformerBlock, TokenAndPositionEmbedding,MultiHeadSelfAttention
import numpy as np
from sklearn.metrics import roc_curve
from Bio.SeqUtils.ProtParam import ProteinAnalysis


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
             'AVP',
             'BBP', 'BIP',
             'CPP', 'DPPIP',
             'QSP', 'SBP', 'THP']


def predict(X_test, X_test_f, y_test, thred, para, h5_model, first_dir):

    print("Prediction is in progress")

    for ii in range(0, len(h5_model)):

        h5_model_path = os.path.join(first_dir, h5_model[ii])
        load_my_model = load_model(h5_model_path, custom_objects={'MultiHeadAttention': MultiHeadAttention,
                                                                  'multilabel_categorical_crossentropy': multilabel_categorical_crossentropy,
                                                                  'MultiHeadSelfAttention':MultiHeadSelfAttention,
                                                                  'TransformerBlock':TransformerBlock,
                                                                  'TokenAndPositionEmbedding':TokenAndPositionEmbedding,
                                                                  })

        # 2.predict
        score = load_my_model.predict(X_test)
        score_label = score
        for i in range(len(score_label)):
            for j in range(len(score_label[i])):
                if score_label[i][j] < 0:  # throld
                    score_label[i][j] = 0
                else:
                    score_label[i][j] = 1


        if ii == 0:
            score_pro = score_label
        else:
            score_pro += score_label
    score_pro = np.array(score_pro)
    score_pro = score_pro / len(h5_model)

    score_label = score_pro
    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < 0.5: # throld
                score_label[i][j] = 0
            else:
                score_label[i][j] = 1

    # evaluation
    aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(score_label, y_test)

    print("Prediction is done")
    print('aiming:', aiming)
    print('coverage:', coverage)
    print('accuracy:', accuracy)
    print('absolute_true:', absolute_true)
    print('absolute_false:', absolute_false)
    print('\n')

    # saving results
    data = []
    data.append('aiming:{}'.format(str(aiming)))
    data.append('coverage:{}'.format(str(coverage)))
    data.append('accuracy:{}'.format(str(accuracy)))
    data.append('absolute_true:{}'.format(str(absolute_true)))
    data.append('absolute_false:{}'.format(str(absolute_false)))
    data.append('\n')
    # with open("TCN/result.txt", 'ab') as x:
    #     np.savetxt(x, np.asarray(data), fmt="%s\t")
    with open("albtion_weight/dropout_result.txt", 'w') as x:
        np.savetxt(x, np.asarray(data), fmt="%s\t")


def test_main(test, para, model_num, modelDir):
    h5_model = []
    # n = 3
    # for i in range(n, n + 1):
    for i in range(1, model_num + 1):
        h5_model.append('model{}.h5'.format('_dropout_' + str(i)))

    # step2:predict
    predict(test[0],test[1], test[2], test[3], para, h5_model,modelDir)

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

def get_features(seq):
    # r1 = list(GetAPseudoAAC(seq,lamda=5).values())
    # from PyBioMed.PyProtein import AAComposition
    # AAC = AAComposition.CalculateDipeptideComposition(seq)
    # r1 = list(AAC.values())
    x  = ProteinAnalysis(seq)
    r2 = [x.gravy()]
    r3 = [x.molecular_weight()]
    r4 = list(x.get_amino_acids_percent().values())
    r5 = [x.charge_at_pH(pH=i) for i in range(14)]
    x6 = list(x.secondary_structure_fraction())
    res = r2 + r3 + r4 + r5 + x6
    return res

first_dir = 'dataset'
max_length = 50

test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')
y_test = np.array(test_sequence_label)
x_test, y_test = PadEncode(test_sequence_data, y_test, max_length)
x_test_f = input_se(test_sequence_data)

x_test = x_test.astype(np.float32)
x_test_f = x_test_f.astype(np.float32)
y_test = y_test.astype(np.float32)

test = [x_test, x_test_f, y_test]
threshold = 0.5
model_num = 10 # model number
test.append(threshold)

ed = 192
ps = 5
fd = 192
dp = 0.5
lr = 0.0015

para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}

modelDir = 'albtion_weight'
test_main(test, para, model_num, modelDir)

