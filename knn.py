from collections import Counter
import numpy as np


def parsing(r_file):
    file_data = np.loadtxt(r_file, delimiter=',')

    label = file_data[0:, 0]
    features = file_data[0:, 1:]

    return features, label


# def k_fold(X,y, k=5):
#     return zip(np.split(X,k), np.split(y,k))
def k_fold(X, k=5):
    offset = int(len(X)/k)
    result_train = list()
    result_test =list()
    for i in range(k):
        test_list = [j for j in range(i*offset,int((i+1)*offset))]
        index_list = [j for j in range(len(X))]
        for t in range(offset):
            index_list.remove(test_list[t])
        result_train.append(index_list)
        result_test.append(test_list)
    return zip(result_train, result_test)

class knnClasifier():
    train_data = np.empty(1)
    train_label = np.empty(1)
    n_neighbor = 5
    distance = "Euclidean"


    def __init__(self,n_neighbor=5, distance="Euclidean"):
        self.n_neighbor = n_neighbor
        self.distance = distance


    def fit(self,X, y):
        self.train_data = X
        self.train_label =y

        return self


    def get_distance(self,X):
        result_dist = list()
        result_indices = list()

        for i in range(len(X)):
            if self.distance == 'Euclidean':
                target_dist = self.euclidean(X[i],self.train_data)
            elif self.distance == 'Manhattan':
                target_dist = self.manhattan(X[i],self.train_data)
            elif self.distance == 'L':
                target_dist = self.L(X[i],self.train_data)
            np_indices = np.argsort(target_dist)[:self.n_neighbor]
            np_dist = np.take(target_dist, np_indices)
            result_dist.append(np_dist)
            result_indices.append(np_indices)

        result_dist = np.array(result_dist)
        result_indices = np.array(result_indices)
        return result_dist, result_indices

    def predict(self,X):
        dist_list, index_list = self.get_distance(X)
        labeling = np.take(self.train_label, index_list)
        result = list()
        for i in labeling:
            counting = Counter(i)
            result.append(counting.most_common(1)[0][0])

        return np.array(result, dtype=int)


    def manhattan(self,X,y):
        dist = np.sum(abs(X-y), axis=1)
        return dist.T
    def L(self,X,y):
        dist = np.amax(abs(y-X), axis=1)
        return dist
    def euclidean(self,X,y):
        dist = np.sqrt(np.sum(np.power(abs(X-y),2),axis=1))
        return dist.T

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


    def prnt_cfm(self):
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