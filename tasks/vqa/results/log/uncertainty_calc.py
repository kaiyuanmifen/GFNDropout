import pickle
import numpy as np
with open('log_run_dp1_cr1_lp1_k0p01_etam2p19_sedim2uc.pkl', 'rb') as a:
    test_result = pickle.load(a,encoding='bytes')
uncertainty = {}
accQA = test_result['accQA']
accAnsType = test_result['accAnsType']
p_value_list = test_result['p_value_list']
p_value_listAnsType = test_result['p_value_listAnsType']
def setUncertainty(accQA, accAnsType, p_value_list, p_value_listAnsType):
    accQA = np.array(accQA)
    p_value_list = np.array(p_value_list)
    uncertainty['overall'] = pavpu_fun(accQA, np.array(p_value_list))
    uncertainty['perAnswerType'] = {ansType: pavpu_fun(
        accAnsType[ansType], np.array(p_value_listAnsType[ansType])) for ansType in accAnsType}
    print(uncertainty)

def pavpu_fun(acc, p_value):
    uc_1 = (p_value > 0.01)
    acc = np.array(acc)
    acc = acc == 1
    pavpu_1 = np.mean(uc_1 * (1 - acc) + (1 - uc_1) * acc)

    uc_2 = (p_value > 0.05)
    pavpu_2 = np.mean(uc_2 * (1 - acc) + (1 - uc_2) * acc)

    uc_3 = (p_value > 0.1)
    pavpu_3 = np.mean(uc_3 * (1 - acc) + (1 - uc_3) * acc)
    print('uncertainty_portion', uc_1.mean(), uc_2.mean(),uc_3.mean())
    return [pavpu_1*100, pavpu_2*100, pavpu_3*100]
setUncertainty(accQA, accAnsType, p_value_list, p_value_listAnsType)



def setUncertainty1(accQA, accAnsType, p_value_list, p_value_listAnsType):
    accQA = np.array(accQA)
    p_value_list = np.array(p_value_list)
    uncertainty['overall'] = pavpu_fun1(accQA, np.array(p_value_list))
    uncertainty['perAnswerType'] = {ansType: pavpu_fun1(
        accAnsType[ansType], np.array(p_value_listAnsType[ansType])) for ansType in accAnsType}
    print(uncertainty)

def pavpu_fun1(acc, p_value):
    uc_1 = (p_value > 0.01)
    acc = np.array(acc)
    acc = acc > 0
    pavpu_1 = np.mean(uc_1 * (1 - acc) + (1 - uc_1) * acc)

    uc_2 = (p_value > 0.05)
    pavpu_2 = np.mean(uc_2 * (1 - acc) + (1 - uc_2) * acc)

    uc_3 = (p_value > 0.1)
    pavpu_3 = np.mean(uc_3 * (1 - acc) + (1 - uc_3) * acc)

    return [pavpu_1, pavpu_2, pavpu_3]
setUncertainty1(accQA, accAnsType, p_value_list, p_value_listAnsType)



def setUncertainty2(accQA, accAnsType, p_value_list, p_value_listAnsType):
    accQA = np.array(accQA)
    p_value_list = np.array(p_value_list)
    uncertainty['overall'] = pavpu_fun2(accQA, np.array(p_value_list))
    uncertainty['perAnswerType'] = {ansType: pavpu_fun2(
        accAnsType[ansType], np.array(p_value_listAnsType[ansType])) for ansType in accAnsType}
    print(uncertainty)

def pavpu_fun2(acc, p_value):
    uc_1 = (p_value > 0.01)
    acc = np.array(acc)
    pavpu_1 = np.mean(uc_1 * (1 - acc) + (1 - uc_1) * acc)

    uc_2 = (p_value > 0.05)
    pavpu_2 = np.mean(uc_2 * (1 - acc) + (1 - uc_2) * acc)

    uc_3 = (p_value > 0.1)
    pavpu_3 = np.mean(uc_3 * (1 - acc) + (1 - uc_3) * acc)

    return [pavpu_1, pavpu_2, pavpu_3]
setUncertainty2(accQA, accAnsType, p_value_list, p_value_listAnsType)


def roc(acc, p_value):
    TPR_list = []
    FPR_list = []
    for i in range(100):
        uc = p_value > i/100.0
        acc = np.array(acc)
        TP = np.sum((1 - uc) * acc)
        FN = np.sum(uc * (1 - acc))
        TN = np.sum(acc * uc)
        FP = np.sum((1 - uc) * (1 - acc))
        TPR_list.append(TP * 1.0/ (TP + FN))
        FPR_list.append(FP * 1.0 / (TP + TN))
    FPR_list = np.array(FPR_list)
    TPR_list = np.array(TPR_list)
    x_axis = FPR_list[1:] - FPR_list[:-1]
    auc = np.sum(x_axis * TPR_list[1:])
    return auc


def setroc(accQA, accAnsType, p_value_list, p_value_listAnsType):
    accQA = np.array(accQA)
    p_value_list = np.array(p_value_list)
    uncertainty['overall'] = roc(accQA, np.array(p_value_list))
    uncertainty['perAnswerType'] = {ansType: roc(
        accAnsType[ansType], np.array(p_value_listAnsType[ansType])) for ansType in accAnsType}
    print(uncertainty)
setroc(accQA, accAnsType, p_value_list, p_value_listAnsType)
