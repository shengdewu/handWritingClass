#coding=utf-8
from files import Files
from kNN import kNN
import os
import numpy as np

def main():
    fop = Files()
    #获取训练数据
    ret, digitsVec, labelVec = fop.txt2mat('digits/trainingDigits')

    knnClass = kNN()
    path = 'digits/testDigits'
    testList = os.listdir(path)
    total = len(testList)
    errCnt = 0
    for index in range(total):
        name = testList[index]
        label = name.split('.')[0][0]
        tmp = path + '/' + name;
        digits = fop.readMat(tmp, 32, 32)
        result = knnClass.classification(digits, digitsVec, labelVec, 3)
        if label != result:
            errCnt += 1.0
    err = errCnt / total
    return


main()