from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math
from pandas.core.frame import DataFrame


# 计算字符熵
def LettersEntropy(str):
    letters, counts = np.unique(list(str), return_counts=True)
    total = sum(counts)
    entropy = 0.0
    for i in range(len(counts)):
        prob = counts[i]/total
        entropy += - prob * math.log(2, prob)
    return entropy

def NumCollect(str):
    total = 0
    for i in str:
        if i.isdigit():
            total +=1
    return total

if __name__ == '__main__':
    # 创建3个特征列表，1个种类列表
    DomainLen = []
    Numbers = []
    Entropy = []
    Type = []
    f = open(r'train.txt')
    for line in f:
        line = line.strip()
        if line == "":
            continue
        tokens = line.split(",")
        domain = tokens[0]
        label = tokens[1]
        DomainLen.append(len(domain))
        Numbers.append(NumCollect(domain))
        Entropy.append(LettersEntropy(domain))
        if label == 'notdga':
            Type.append(0)
        else:
            Type.append(1)
    traindata = {'Length': DomainLen, 'NumInDomain': Numbers, 'Entropy': Entropy, 'Type': Type}
    traindata = DataFrame(traindata)
    y = traindata.Type
    x = traindata.drop('Type', axis=1)
    xtrain = x
    ytrain = y
    testDomainLen = []
    testNumbers = []
    testEntropy = []
    testType = []
    testDomainName = []
    TestFile = open(r'test.txt')
    for line in TestFile:
        line = line.strip()
        if line == "":
            continue
        testDomainName.append(line)
        testDomainLen.append(len(line))
        testNumbers.append(NumCollect(line))
        testEntropy.append(LettersEntropy(line))
    testdata = {'Length': testDomainLen, 'NumInDomain': testNumbers, 'Entropy': testEntropy}
    testdata = DataFrame(testdata)
    # 设置测试数据集并实例化
    xtest = testdata
    rfc = RandomForestClassifier()
    # 用训练集数据训练模型
    rfc = rfc.fit(xtrain, ytrain)
    ytest = rfc.predict(xtest)
    for i in range(len(ytest)):
        if ytest[i] == 0:
            testType.append('notdga')
        else:
            testType.append('dga')
    with open('result.txt', 'w+') as ResultFile:
        for i in range(len(testDomainName)):
            line = testDomainName[i] + ',' + testType[i] + '\n'
            ResultFile.write(line)
