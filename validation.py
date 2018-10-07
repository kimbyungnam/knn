import numpy as np

class validation():
    predict = list()
    label = list()
    confusion_metric = list()
    max_len = 0

    def __init__(self, predict, label, max_len):
        self.predict = predict
        self.label = label
        self.max_len = max_len
        self.confusion_metric = np.zeros((max_len,max_len), dtype=int)
        for i in range(len(predict)):
            self.confusion_metric[int(self.label[i]), int(self.predict[i])] += 1


    def cfm(self):
        print self.confusion_metric


    def recall(self):
        vector = np.ones(self.max_len)
        denominator = np.dot(self.confusion_metric, vector)
        numerator = np.diag(self.confusion_metric)
        pre_list = numerator/denominator
        result = np.sum(pre_list)/self.max_len

        return result


    def precision(self):
        vector = np.ones(self.max_len)
        denominator = np.dot(self.confusion_metric.T, vector)
        numerator = np.diag(self.confusion_metric)
        pre_list = numerator/denominator
        result = np.sum(pre_list)/self.max_len

        return result

    def F1_measure(self):
        prec = self.precision()
        rec = self.recall()

        return 2*((prec*rec)/(prec+rec))

    def accuracy(self):
        denominator = np.sum(self.confusion_metric, dtype=float)
        numerator = np.sum(np.diag(self.confusion_metric), dtype=float)

        return numerator/denominator