import numpy as np

class kNN(object):
    def __init__(self):
        print("kNN create kNN")
        return

#input 1 dims
#sample numpy array
#label numpy array
    def classification(self, input, sample, label, k):
        rows = sample.shape[0]
        inputX = np.tile(input, (rows, 1))
        dis = self.distance(inputX, sample)
        index = np.argsort(dis)

        maxLabelCnt = index.shape[0] if index.shape[0] > 1 else index.shape[1] ;
        k = k if k<maxLabelCnt else maxLabelCnt

        sortLabel = {}
        for i in range(k):
            lb = label[index[i]]
            sortLabel[lb] = sortLabel.get(lb, 0) + 1
        result = sorted(sortLabel.items(), key=lambda item:item[1], reverse = True)
        return result[0][0]

    def distance(self, src, dst):
        diff = src - dst
        diff = diff **2
        value = diff.sum(axis=1)
        value = np.sqrt(value)
        return value