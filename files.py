#coding=utf-8
import numpy as np
import os

class Files(object):
    #path 内容的根目录
    #ret 0 成功； -1失败
    #digitsVec 数字向量
    #labelVec 每行对应的标签
    def txt2mat(self, path):
        fileList = os.listdir(path)
        rows = 32
        cols = 32
        labelCnt = len(fileList)
        digitsVec = np.zeros((labelCnt, rows * cols))
        labelVec = []
        for index in range(labelCnt):
            name = fileList[index]
            label = name.split('.')[0][0]
            labelVec.append(label)
            tmp = path + '/' + name;
            digitsVec[index] = self.readMat(tmp, rows, cols)
        if labelCnt != len(labelVec):
            return -1, digitsVec, labelVec
        return 0, digitsVec, labelVec

    def readMat(self, path, rows, cols):
        f = open(path, 'r')
        vec = np.zeros(rows * cols)
        for row in range(rows):
            line = f.readline()
            line = line.strip()
            for col in range(cols):
                vec[row * rows + col] = int(line[col])
        return vec

